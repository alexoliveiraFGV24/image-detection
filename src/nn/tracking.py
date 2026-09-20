"""
Associacao e gestao de tracks (nascimento / morte), de autoria propria --
o enunciado proibe SORT, DeepSORT, ByteTrack e afins.

O rastreador e o mesmo em todas as partes; o que muda entre elas e o
MODELO DE MOVIMENTO que fornece a caixa prevista de cada track para o
quadro seguinte:

    StaticMotion    Parte 1: a previsao e a ultima caixa observada.
                    E a "associacao ingenua" do enunciado.
    KalmanMotion    filtro de Kalman de velocidade constante. PERMITIDO
                    apenas como baseline de comparacao.
    RNNMotion       Parte 2, Trilha A: quem carrega o estado e a rede
                    recorrente de src/nn/models.py (MotionModel).

Regra de associacao (documentada aqui porque regras diferentes dao
numeros diferentes):

    1. a cada quadro, IoU entre a caixa PREVISTA de cada track viva e
       cada deteccao do quadro (deteccoes com score < min_score saem);
    2. matching guloso por IoU decrescente (ou Hungarian), limiar fixo
       `iou_threshold`;
    3. deteccao sem par -> track NOVA (id novo, nunca reutilizado);
    4. track sem par -> "coasting": o estado roda para frente sem
       observacao e a idade cresce; com idade > `max_age` a track morre.
    5. so tracks OBSERVADAS no quadro entram na saida (uma track em
       coasting nao emite caixa) -- assim as metricas de identidade
       nao sao contaminadas por caixas inventadas.
"""

import numpy as np

from src.nn.boxes import (
    X1, Y2, CONF,
    box_iou_matrix, split_by_frame, make_table, empty_table,
    xyxy_to_cxcywh, cxcywh_to_xyxy,
)
from src.nn.metrics import MATCHERS


# ============================================================
# MODELOS DE MOVIMENTO  (interface)
# ============================================================


class MotionModelBase:
    """
    Interface minima que o Tracker exige. O `state` e opaco para o
    rastreador; so o modelo de movimento sabe o que ele contem.

        init(box, score, t)      -> state       (nascimento da track)
        predict(state)           -> box (4,)    (caixa esperada no quadro atual)
        update(state, box, score, t) -> state   (observacao recebida)
        coast(state, t)          -> state       (quadro sem observacao)

    `predict` e um getter: `update`/`coast` ja deixam pronta a previsao
    para o proximo quadro. Isso vale para os tres modelos -- na RNN, o
    passo da celula acontece em update/coast, e a previsao e a saida.
    """

    def init(self, box, score, t):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def update(self, state, box, score, t):
        raise NotImplementedError

    def coast(self, state, t):
        raise NotImplementedError


class StaticMotion(MotionModelBase):
    """Parte 1: a caixa prevista e a ultima caixa observada."""

    def init(self, box, score, t):
        return {"box": np.asarray(box, dtype=np.float64).copy()}

    def predict(self, state):
        return state["box"]

    def update(self, state, box, score, t):
        state["box"] = np.asarray(box, dtype=np.float64).copy()
        return state

    def coast(self, state, t):
        return state


class KalmanMotion(MotionModelBase):
    """
    Filtro de Kalman linear de velocidade constante sobre
    (cx, cy, w, h, vx, vy, vw, vh). Baseline honesto, escrito aqui do
    zero; NAO e o modelo temporal da Parte 2.

    Args:
        process_noise: desvio do ruido de processo (posicao e tamanho);
            a velocidade usa process_noise / 10.
        measurement_noise: desvio do ruido de medida (pixels).
    """

    def __init__(self, process_noise=1.0, measurement_noise=1.0, velocity_decay=1.0):

        self.F = np.eye(8)
        self.F[0:4, 4:8] = np.eye(4)            # x += v * dt (dt = 1 quadro)
        self.F[4:8, 4:8] *= velocity_decay

        self.H = np.zeros((4, 8))
        self.H[0:4, 0:4] = np.eye(4)

        q = np.array([process_noise] * 4 + [process_noise / 10] * 4) ** 2
        self.Q = np.diag(q)
        self.R = np.eye(4) * measurement_noise ** 2

    @staticmethod
    def _to_measurement(box):
        return xyxy_to_cxcywh(np.asarray(box, dtype=np.float64))

    def _project(self, state):
        """Passo de predicao: x <- F x, P <- F P F^T + Q."""

        state["x"] = self.F @ state["x"]
        state["P"] = self.F @ state["P"] @ self.F.T + self.Q
        return state

    def init(self, box, score, t):

        x = np.zeros(8)
        x[0:4] = self._to_measurement(box)

        P = np.eye(8) * 10.0
        P[4:8, 4:8] *= 100.0            # velocidade desconhecida no inicio

        state = {"x": x, "P": P}
        return self._project(state)

    def predict(self, state):
        return cxcywh_to_xyxy(state["x"][0:4])

    def update(self, state, box, score, t):

        z = self._to_measurement(box)

        y = z - self.H @ state["x"]
        S = self.H @ state["P"] @ self.H.T + self.R
        K = state["P"] @ self.H.T @ np.linalg.inv(S)

        state["x"] = state["x"] + K @ y
        state["P"] = (np.eye(8) - K @ self.H) @ state["P"]

        return self._project(state)

    def coast(self, state, t):
        return self._project(state)


class RNNMotion(MotionModelBase):
    """
    Adaptador para o MotionModel recorrente (src/nn/models.py) da
    Parte 2 -- Trilha A. Um estado recorrente POR TRACK.

    A cada quadro a celula recebe a ultima observacao (caixa, confianca,
    dt, flag de observado) e preve a caixa do quadro seguinte. Sob
    oclusao (`coast`) a entrada e a propria previsao anterior, com a
    flag de observado em zero: o estado roda para frente sem observacao.

    Args:
        model: MotionModel treinado.
        image_size: (largura, altura) para normalizar as caixas.
        device: torch device.
    """

    def __init__(self, model, image_size, device="cpu"):

        import torch

        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = np.asarray(image_size, dtype=np.float64)

    def _normalize(self, box):
        cxcywh = xyxy_to_cxcywh(np.asarray(box, dtype=np.float64))
        scale = np.concatenate([self.image_size, self.image_size])
        return cxcywh / scale

    def _denormalize(self, cxcywh):
        scale = np.concatenate([self.image_size, self.image_size])
        return cxcywh_to_xyxy(np.asarray(cxcywh, dtype=np.float64) * scale)

    def _step(self, state, box_n, score, dt, observed):

        torch = self.torch

        with torch.no_grad():

            box_t = torch.as_tensor(box_n, dtype=torch.float32, device=self.device)[None]
            prev_t = torch.as_tensor(state["last_input"], dtype=torch.float32, device=self.device)[None]

            out = self.model.step(
                box=box_t,
                prev_box=prev_t,
                conf=torch.full((1, 1), float(score), device=self.device),
                dt=torch.full((1, 1), float(dt), device=self.device),
                observed=torch.full((1, 1), float(observed), device=self.device),
                state=state["h"],
            )

        state["h"] = out["state"]
        state["last_input"] = box_n
        state["pred_n"] = out["box"][0].cpu().numpy()
        state["log_var"] = out["log_var"][0].cpu().numpy() if out.get("log_var") is not None else None
        return state

    def init(self, box, score, t):

        box_n = self._normalize(box)

        state = {
            "h": self.model.init_state(1, self.device),
            "last_input": box_n,
            "t_last": t,
        }

        return self._step(state, box_n, score, dt=1.0, observed=1.0)

    def predict(self, state):
        return self._denormalize(state["pred_n"])

    def update(self, state, box, score, t):

        dt = float(t - state["t_last"])
        state["t_last"] = t

        return self._step(state, self._normalize(box), score, dt=dt, observed=1.0)

    def coast(self, state, t):

        # Alimenta a propria previsao (free-running), sem observacao
        return self._step(state, state["pred_n"], score=0.0, dt=1.0, observed=0.0)


# ============================================================
# TRACKER
# ============================================================


class Track:
    """Uma trajetoria viva: id, estado do modelo de movimento, idade."""

    __slots__ = ("id", "state", "age", "hits", "start", "last_box", "last_score")

    def __init__(self, track_id, state, t, box, score):
        self.id = track_id
        self.state = state
        self.age = 0            # quadros consecutivos sem observacao
        self.hits = 1           # observacoes recebidas
        self.start = t
        self.last_box = np.asarray(box, dtype=np.float64)
        self.last_score = float(score)


class Tracker:
    """
    Rastreamento por associacao quadro a quadro (regras no cabecalho).

    Args:
        motion: instancia de MotionModelBase (StaticMotion por padrao).
        iou_threshold: limiar fixo de associacao.
        max_age: quadros sem observacao antes de a track morrer.
        matching: "greedy" ou "hungarian".
        min_score: deteccoes abaixo disto sao ignoradas.
        min_hits: uma track so aparece na saida depois de `min_hits`
            observacoes (1 = aparece ja no nascimento).
    """

    def __init__(self, motion=None, iou_threshold=0.3, max_age=5, matching="greedy",
                 min_score=0.0, min_hits=1):

        self.motion = motion if motion is not None else StaticMotion()
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.matching = matching
        self.min_score = min_score
        self.min_hits = min_hits

        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.history = []       # linhas (frame, id, x1, y1, x2, y2, score)
        self.predictions = []   # (frame, id, caixa prevista) -- para figuras

    # --------------------------------------------------------

    def update(self, t, boxes, scores=None):
        """
        Processa um quadro.

        Args:
            t: indice do quadro.
            boxes: (N, 4) deteccoes xyxy.
            scores: (N,) confiancas (1 se None).

        Returns:
            tabela (M, 7) das tracks observadas neste quadro.
        """

        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.ones(len(boxes)) if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)

        keep = scores >= self.min_score
        boxes, scores = boxes[keep], scores[keep]

        # 1. caixas previstas das tracks vivas
        predicted = np.array(
            [self.motion.predict(tr.state) for tr in self.tracks],
            dtype=np.float64,
        ).reshape(-1, 4)

        for tr, box in zip(self.tracks, predicted):
            self.predictions.append((t, tr.id, box.copy()))

        # 2. associacao
        iou = box_iou_matrix(predicted, boxes)
        pairs = MATCHERS[self.matching](iou, self.iou_threshold)

        matched_tracks = set()
        matched_dets = set()

        rows = []

        for i, j in pairs:

            tr = self.tracks[i]
            tr.state = self.motion.update(tr.state, boxes[j], scores[j], t)
            tr.age = 0
            tr.hits += 1
            tr.last_box = boxes[j]
            tr.last_score = scores[j]

            matched_tracks.add(i)
            matched_dets.add(j)

            if tr.hits >= self.min_hits:
                rows.append((t, tr.id, *boxes[j], scores[j]))

        # 3. tracks sem par: coasting / morte
        survivors = []

        for i, tr in enumerate(self.tracks):

            if i in matched_tracks:
                survivors.append(tr)
                continue

            tr.age += 1

            if tr.age <= self.max_age:
                tr.state = self.motion.coast(tr.state, t)
                survivors.append(tr)

        self.tracks = survivors

        # 4. deteccoes sem par: nascimento
        for j in range(len(boxes)):

            if j in matched_dets:
                continue

            state = self.motion.init(boxes[j], scores[j], t)
            tr = Track(self.next_id, state, t, boxes[j], scores[j])
            self.next_id += 1
            self.tracks.append(tr)

            if tr.hits >= self.min_hits:
                rows.append((t, tr.id, *boxes[j], scores[j]))

        self.history.extend(rows)

        return np.asarray(rows, dtype=np.float64).reshape(-1, 7)

    # --------------------------------------------------------

    def run(self, detections, frames=None):
        """
        Roda o rastreador numa tabela de deteccoes (N, 7) inteira.

        Args:
            detections: tabela (frame, id(-1), x1, y1, x2, y2, score).
            frames: lista de quadros a processar (padrao: todos os que
                aparecem na tabela, em ordem). Quadros sem deteccao
                tambem sao processados (as tracks envelhecem).

        Returns:
            tabela (M, 7) de trajetorias previstas.
        """

        self.reset()

        by_frame = split_by_frame(detections)

        if frames is None:
            if not by_frame:
                return empty_table()
            frames = range(min(by_frame), max(by_frame) + 1)

        for t in frames:

            rows = by_frame.get(int(t))

            if rows is None:
                self.update(int(t), np.zeros((0, 4)), np.zeros(0))
            else:
                self.update(int(t), rows[:, X1:Y2 + 1], rows[:, CONF])

        return self.result()

    def result(self):
        """Tabela (M, 7) acumulada ate agora."""

        if not self.history:
            return empty_table()

        return np.asarray(self.history, dtype=np.float64).reshape(-1, 7)

    def predicted_boxes_table(self):
        """
        Tabela (K, 7) com a caixa PREVISTA pelo modelo de movimento para
        cada track viva em cada quadro (conf = 1). Serve para as figuras
        da Parte 4 (caixa prevista pela recorrencia vs. observada).
        """

        if not self.predictions:
            return empty_table()

        frames = [p[0] for p in self.predictions]
        ids = [p[1] for p in self.predictions]
        boxes = [p[2] for p in self.predictions]

        return make_table(frames, ids, boxes, np.ones(len(frames)))
