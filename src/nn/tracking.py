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
       `iou_threshold` -- ou, com `gate`, o portao adaptativo: tambem
       entram pares com IoU abaixo do limiar se a deteccao cair dentro
       da elipse de incerteza prevista pela rede (Trilha A, opcional);
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
    Interface que o Tracker exige. O `state` e opaco para o rastreador;
    so o modelo de movimento sabe o que ele contem.

        init(box, score, t)      -> state       (nascimento da track)
        predict(state)           -> box (4,)    (caixa esperada no quadro atual)
        update(state, box, score, t) -> state   (observacao recebida)
        coast(state, t)          -> state       (quadro sem observacao)

    `predict` e um getter: `update`/`coast` ja deixam pronta a previsao
    para o proximo quadro. Isso vale para os tres modelos -- na RNN, o
    passo da celula acontece em update/coast, e a previsao e a saida.

    O Tracker chama as versoes EM LOTE (`*_many`, todas as tracks do
    quadro de uma vez). Aqui elas so repetem a versao unitaria; a RNN as
    sobrescreve para rodar um passo da celula para todas as tracks numa
    unica chamada.
    """

    def init(self, box, score, t):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def update(self, state, box, score, t):
        raise NotImplementedError

    def coast(self, state, t):
        raise NotImplementedError

    # --- versoes em lote

    def init_many(self, boxes, scores, t):
        return [self.init(b, s, t) for b, s in zip(boxes, scores)]

    def predict_many(self, states):
        return np.asarray([self.predict(s) for s in states], dtype=np.float64).reshape(-1, 4)

    def update_many(self, states, boxes, scores, t):
        return [self.update(st, b, s, t) for st, b, s in zip(states, boxes, scores)]

    def coast_many(self, states, t):
        return [self.coast(st, t) for st in states]

    def gate_distance_many(self, states, boxes):
        """Distancia de Mahalanobis (tracks x deteccoes) para o portao adaptativo."""
        raise NotImplementedError(f"{type(self).__name__} nao preve incerteza")


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
        velocity_decay: fator aplicado a velocidade a cada passo (1 =
            velocidade constante).
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
    Parte 2 -- Trilha A. Um estado recorrente POR TRACK, mas um unico
    passo da celula por quadro para todas as tracks (lote).

    A cada quadro a celula recebe a ultima observacao (a caixa da
    deteccao associada, com a flag de observado = 1) e preve a caixa do
    quadro seguinte. Sob oclusao (`coast`) a entrada e a propria previsao
    anterior, com a flag em zero: o estado roda para frente sem
    observacao, e a track sobrevive enquanto a previsao continuar
    encontrando a pessoa (ou ate `max_age`).

    Caixas dentro do modelo: (cx, cy, w, h) divididas pela ALTURA da
    imagem (mesma escala nos dois eixos). dt: quadros entre dois passos,
    em unidades de 1/30 s (dt = 30 / fps a cada quadro).

    Args:
        model: MotionModel treinado.
        image_height: altura da imagem da sequencia (px).
        fps: taxa de quadros da sequencia.
        device: torch device.
    """

    def __init__(self, model, image_height, fps=30.0, device="cpu"):

        import torch

        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.scale = float(image_height)
        self.dt_unit = 30.0 / float(fps)

    # --- conversoes

    def _to_model(self, boxes_xyxy):
        return xyxy_to_cxcywh(np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)) / self.scale

    def _to_pixels(self, cxcywh):
        return cxcywh_to_xyxy(np.asarray(cxcywh, dtype=np.float64).reshape(-1, 4) * self.scale)

    def _tensor(self, array):
        return self.torch.as_tensor(np.asarray(array), dtype=self.torch.float32, device=self.device)

    # --- um passo da celula para varias tracks

    def _run(self, states, box_in, observed, t):

        from src.nn.models import stack_states, unstack_state

        if len(states) == 0:
            return states

        torch = self.torch

        prev_in = np.stack([s["last_input"] for s in states])
        dt = np.array([(t - s["t_step"]) * self.dt_unit for s in states])

        with torch.no_grad():
            out = self.model.step(
                box_in=self._tensor(box_in),
                prev_in=self._tensor(prev_in),
                observed=self._tensor(np.asarray(observed, dtype=np.float64).reshape(-1, 1)),
                state=stack_states([s["h"] for s in states]),
                dt=self._tensor(dt.reshape(-1, 1)),
            )

        pred = out["box"].cpu().numpy().astype(np.float64)
        log_var = out["log_var"].cpu().numpy().astype(np.float64) if out["log_var"] is not None else None

        for i, s in enumerate(states):
            s["h"] = unstack_state(out["state"], i)
            s["last_input"] = np.asarray(box_in[i], dtype=np.float64)
            s["pred"] = pred[i]
            s["log_var"] = log_var[i] if log_var is not None else None
            s["t_step"] = t

        return states

    def init_many(self, boxes, scores, t):

        boxes_n = self._to_model(boxes)

        states = [{
            "h": self.model.init_state(1, self.device),
            "last_input": b,                # sem passado: velocidade de entrada zero
            "t_step": t - 1,                # primeiro passo com dt nominal
        } for b in boxes_n]

        return self._run(states, boxes_n, np.ones(len(states)), t)

    def predict_many(self, states):

        if len(states) == 0:
            return np.zeros((0, 4))

        return self._to_pixels(np.stack([s["pred"] for s in states]))

    def update_many(self, states, boxes, scores, t):
        return self._run(states, self._to_model(boxes), np.ones(len(states)), t)

    def coast_many(self, states, t):

        if len(states) == 0:
            return states

        # free-running: a entrada e a propria previsao, sem observacao (ou a
        # ultima caixa observada, congelada, na sonda de memoria da Parte 3)
        if getattr(self.model, "coast_input", "prediction") == "last_observation":
            own = np.stack([s["last_input"] for s in states])
        else:
            own = np.stack([s["pred"] for s in states])

        return self._run(states, own, np.zeros(len(states)), t)

    def gate_distance_many(self, states, boxes):
        """
        Distancia de Mahalanobis entre cada deteccao e a previsao de cada
        track, no espaco dos deltas (src/nn/models.py::encode_delta), com
        a variancia que a rede preve. Sob oclusao longa a rede "abre" a
        variancia e o portao cresce junto -- o portao adaptativo.
        """

        from src.nn.models import encode_delta

        if self.model.predict_uncertainty is False:
            raise NotImplementedError("o MotionModel foi treinado sem a cabeca de incerteza")

        torch = self.torch
        n_tracks, n_dets = len(states), len(boxes)

        if n_tracks == 0 or n_dets == 0:
            return np.zeros((n_tracks, n_dets))

        ref = self._tensor(np.stack([s["last_input"] for s in states]))           # (N, 4)
        pred = self._tensor(np.stack([s["pred"] for s in states]))                # (N, 4)
        log_var = self._tensor(np.stack([s["log_var"] for s in states]))          # (N, 4)
        dets = self._tensor(self._to_model(boxes))                                 # (M, 4)

        with torch.no_grad():
            delta_pred = encode_delta(pred, ref)                                   # (N, 4)
            delta_obs = encode_delta(dets[None, :, :], ref[:, None, :])            # (N, M, 4)
            d2 = ((delta_obs - delta_pred[:, None, :]) ** 2 * torch.exp(-log_var)[:, None, :]).sum(-1)

        return d2.cpu().numpy().astype(np.float64)

    # --- versoes unitarias (por completude)

    def init(self, box, score, t):
        return self.init_many([box], [score], t)[0]

    def predict(self, state):
        return self.predict_many([state])[0]

    def update(self, state, box, score, t):
        return self.update_many([state], [box], [score], t)[0]

    def coast(self, state, t):
        return self.coast_many([state], t)[0]


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
        min_score: deteccoes abaixo disto sao ignoradas (None = nenhuma;
            o DPM do MOT17 tem scores negativos).
        min_hits: uma track so aparece na saida depois de `min_hits`
            observacoes (1 = aparece ja no nascimento).
        gate: None (so IoU) ou o limiar da distancia de Mahalanobis do
            portao adaptativo (ex.: 9.49 = qui-quadrado com 4 graus de
            liberdade a 95 %). Exige um modelo de movimento com incerteza.
    """

    def __init__(self, motion=None, iou_threshold=0.3, max_age=5, matching="greedy",
                 min_score=None, min_hits=1, gate=None):

        self.motion = motion if motion is not None else StaticMotion()
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.matching = matching
        self.min_score = min_score
        self.min_hits = min_hits
        self.gate = gate

        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.history = []       # linhas (frame, id, x1, y1, x2, y2, score)
        self.predictions = []   # (frame, id, caixa prevista, idade) -- para figuras

    # --------------------------------------------------------

    def _association_scores(self, predicted, boxes):
        """
        Matriz de "afinidade" (tracks x deteccoes) e o limiar para o
        matcher. Sem portao: o proprio IoU e o limiar fixo. Com portao: o
        IoU onde o par e admissivel -- IoU >= limiar OU deteccao dentro
        da elipse de incerteza da track -- e zero fora; pares admitidos so
        pelo portao recebem pelo menos 1e-3, para o matcher considera-los.
        """

        iou = box_iou_matrix(predicted, boxes)

        if self.gate is None or iou.size == 0:
            return iou, self.iou_threshold

        d2 = self.motion.gate_distance_many([tr.state for tr in self.tracks], boxes)
        allowed = (iou >= self.iou_threshold) | (d2 <= self.gate)

        return np.where(allowed, np.maximum(iou, 1e-3), 0.0), 1e-3

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

        keep = np.ones(len(scores), dtype=bool) if self.min_score is None else scores >= self.min_score
        boxes, scores = boxes[keep], scores[keep]

        # 1. caixas previstas das tracks vivas
        predicted = self.motion.predict_many([tr.state for tr in self.tracks])

        for tr, box in zip(self.tracks, predicted):
            self.predictions.append((t, tr.id, box.copy(), tr.age))

        # 2. associacao
        affinity, threshold = self._association_scores(predicted, boxes)
        pairs = MATCHERS[self.matching](affinity, threshold)

        matched_tracks = {i for i, _ in pairs}
        matched_dets = {j for _, j in pairs}

        rows = []

        if pairs:
            ti = [i for i, _ in pairs]
            dj = [j for _, j in pairs]
            new_states = self.motion.update_many([self.tracks[i].state for i in ti], boxes[dj], scores[dj], t)

            for i, j, state in zip(ti, dj, new_states):
                tr = self.tracks[i]
                tr.state = state
                tr.age = 0
                tr.hits += 1
                tr.last_box = boxes[j]
                tr.last_score = scores[j]

                if tr.hits >= self.min_hits:
                    rows.append((t, tr.id, *boxes[j], scores[j]))

        # 3. tracks sem par: coasting / morte
        coasting = []
        survivors = []

        for i, tr in enumerate(self.tracks):

            if i in matched_tracks:
                survivors.append(tr)
                continue

            tr.age += 1

            if tr.age <= self.max_age:
                coasting.append(tr)
                survivors.append(tr)

        if coasting:
            for tr, state in zip(coasting, self.motion.coast_many([tr.state for tr in coasting], t)):
                tr.state = state

        self.tracks = survivors

        # 4. deteccoes sem par: nascimento
        new_dets = [j for j in range(len(boxes)) if j not in matched_dets]

        if new_dets:
            states = self.motion.init_many(boxes[new_dets], scores[new_dets], t)

            for j, state in zip(new_dets, states):
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
        cada track viva em cada quadro; a coluna `conf` guarda a IDADE da
        track naquele quadro (0 = observada no quadro anterior, k = k
        quadros de coasting). Serve para as figuras da caixa prevista pela
        recorrencia vs. a observada.
        """

        if not self.predictions:
            return empty_table()

        frames = [p[0] for p in self.predictions]
        ids = [p[1] for p in self.predictions]
        boxes = [p[2] for p in self.predictions]
        ages = [p[3] for p in self.predictions]

        return make_table(frames, ids, boxes, ages)


def track_sequence(detections, n_frames, motion=None, return_tracker=False, **tracker_kwargs):
    """
    Atalho: roda um Tracker novo em TODOS os quadros 0..n_frames-1 (os
    quadros sem nenhuma deteccao tambem contam: as tracks envelhecem).

    Args:
        detections: tabela (N, 7) de deteccoes.
        n_frames: numero de quadros da sequencia.
        motion: modelo de movimento (padrao: StaticMotion, a Parte 1).
        return_tracker: devolve tambem o Tracker (para as previsoes).
        **tracker_kwargs: iou_threshold, max_age, matching, gate, ...

    Returns:
        tabela (M, 7) de trajetorias previstas (e o Tracker, se pedido).
    """

    tracker = Tracker(motion if motion is not None else StaticMotion(), **tracker_kwargs)
    result = tracker.run(detections, frames=range(n_frames))

    return (result, tracker) if return_tracker else result
