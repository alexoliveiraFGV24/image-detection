"""
Trajetorias do ground truth como sequencias de treino da Trilha A (Parte 2).

O MotionModel e treinado nas trajetorias do GT, mas na inferencia ele ve
outra coisa: as caixas do DETECTOR (com erro), so quando o detector acha
a pessoa, e a propria previsao quando nao acha. O treino reproduz isso:

    alvo     a caixa VERDADEIRA (o MOT17 anota a pessoa inteira mesmo
             ocluida -- o modelo aprende a seguir quem nao se ve);
    entrada  DetectorReplay: a caixa do PROPRIO detector casada com a
             pessoa naquele quadro (erro, falhas e correlacao temporal
             reais) -- ou DetectionNoise: a caixa verdadeira estragada
             por um ruido gaussiano com os desvios medidos;
    observado  1 onde o detector achou a pessoa (ou, no DetectionNoise,
             sorteado pela visibilidade), mais blocos de ausencia
             simulados -- oclusoes longas, para o estado aprender a rodar
             sem observacao.

Convencoes: caixas (cx, cy, w, h) divididas pela ALTURA da imagem (mesma
escala nos dois eixos: preserva a razao de aspecto); dt = quadros entre
dois passos em unidades de 1/30 s (dt = 30 / fps a cada quadro: o
MOT17-05 roda a 14 fps, o MOT17-13 a 25).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.nn.boxes import FRAME, X1, Y2, CONF, split_by_id, xyxy_to_cxcywh, cxcywh_to_xyxy


REFERENCE_FPS = 30.0


def normalize_boxes(boxes_xyxy, image_height):
    """(N, 4) xyxy em pixels -> (N, 4) cxcywh divididas pela altura da imagem."""

    return xyxy_to_cxcywh(np.asarray(boxes_xyxy, dtype=np.float64)) / float(image_height)


def denormalize_boxes(cxcywh, image_height):
    """Inversa de normalize_boxes."""

    return cxcywh_to_xyxy(np.asarray(cxcywh, dtype=np.float64) * float(image_height))


def extract_trajectories(table, image_height, fps=REFERENCE_FPS, min_length=2, sequence=None):
    """
    Uma trajetoria por identidade do GT: dict com frames (L,), boxes (L, 4)
    normalizadas, visibility (L,), dt_unit (= 30 / fps) e o nome da sequencia.
    """

    trajectories = []

    for track_id, rows in split_by_id(table).items():

        if len(rows) < min_length:
            continue

        trajectories.append({
            "sequence": sequence,
            "id": int(track_id),
            "frames": rows[:, FRAME].astype(int),
            "boxes": normalize_boxes(rows[:, X1:Y2 + 1], image_height),
            "visibility": rows[:, CONF].astype(np.float64),
            "dt_unit": REFERENCE_FPS / float(fps),
        })

    return trajectories


def attach_detections(trajectories, gt_table, detections, image_height, iou_threshold=0.5):
    """
    Para cada trajetoria, a caixa do detector casada com a pessoa em cada
    quadro (Hungarian por quadro entre o GT e as deteccoes, IoU >= limiar)
    -- NaN onde o detector nao a achou. Guarda em traj["detections"].
    """

    from src.nn.boxes import ID, split_by_frame, box_iou_matrix
    from src.nn.metrics import hungarian_match

    det_by_frame = split_by_frame(detections)
    lookup = {}

    for frame, g in split_by_frame(gt_table).items():

        d = det_by_frame.get(frame)

        if d is None:
            continue

        iou = box_iou_matrix(g[:, X1:Y2 + 1], d[:, X1:Y2 + 1])

        for i, j in hungarian_match(iou, iou_threshold):
            lookup[(int(g[i, ID]), int(frame))] = d[j, X1:Y2 + 1]

    for traj in trajectories:

        boxes = np.full((len(traj["frames"]), 4), np.nan)

        for k, f in enumerate(traj["frames"]):
            box = lookup.get((traj["id"], int(f)))
            if box is not None:
                boxes[k] = normalize_boxes(box[None], image_height)[0]

        traj["detections"] = boxes

    return trajectories


class DetectionNoise:
    """
    Como o detector estraga uma trajetoria verdadeira.

    Args:
        sigma: desvio do erro da caixa, relativo ao tamanho:
            (dx / w, dy / h, dlog w, dlog h).
        visibility_bins, p_detect: P(detectado | visibilidade) por faixa
            (bins com len(p_detect) + 1 bordas). None = sempre detectado.
        occlusion_prob: probabilidade de inserir um bloco de ausencia
            numa janela; occlusion_len: (min, max) do bloco, em quadros.
        outlier_prob: probabilidade, por quadro observado, de a caixa
            "detectada" ser de OUTRA pessoa (associacao errada no
            rastreador): deslocada de 0,3 a 1,0 largura numa direcao
            qualquer e com escala sorteada em +-20 %.
    """

    def __init__(self, sigma=(0.06, 0.025, 0.08, 0.045), visibility_bins=None, p_detect=None,
                 occlusion_prob=0.3, occlusion_len=(1, 40), outlier_prob=0.0):

        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.visibility_bins = None if visibility_bins is None else np.asarray(visibility_bins, dtype=np.float64)
        self.p_detect = None if p_detect is None else np.asarray(p_detect, dtype=np.float64)
        self.occlusion_prob = occlusion_prob
        self.occlusion_len = occlusion_len
        self.outlier_prob = outlier_prob

    def detection_probability(self, visibility):

        if self.p_detect is None:
            return np.ones_like(visibility)

        index = np.clip(np.searchsorted(self.visibility_bins, visibility, side="right") - 1, 0, len(self.p_detect) - 1)
        return self.p_detect[index]

    def sample(self, boxes, visibility, rng, detections=None):
        """
        Args:
            boxes: (L, 4) cxcywh verdadeiras; visibility: (L,).
            detections: ignorado (ver DetectorReplay).

        Returns:
            inputs (L, 4) caixas "detectadas" e observed (L,) bool.
        """

        L = len(boxes)

        observed = rng.random(L) < self.detection_probability(visibility)

        if self.occlusion_prob > 0 and L > 2 and rng.random() < self.occlusion_prob:
            lo, hi = self.occlusion_len
            n = int(rng.integers(lo, hi + 1))
            s = int(rng.integers(1, max(2, L - 1)))
            observed[s:s + n] = False

        observed[0] = True              # a track nasce de uma deteccao

        noise = rng.standard_normal((L, 4)) * self.sigma

        inputs = boxes.copy()
        inputs[:, 0] += noise[:, 0] * boxes[:, 2]
        inputs[:, 1] += noise[:, 1] * boxes[:, 3]
        inputs[:, 2] *= np.exp(noise[:, 2])
        inputs[:, 3] *= np.exp(noise[:, 3])

        if self.outlier_prob > 0:
            wrong = (rng.random(L) < self.outlier_prob) & observed
            wrong[0] = False
            n = int(wrong.sum())
            if n:
                angle = rng.uniform(0, 2 * np.pi, n)
                dist = rng.uniform(0.3, 1.0, n) * boxes[wrong, 2]
                inputs[wrong, 0] += dist * np.cos(angle)
                inputs[wrong, 1] += dist * np.sin(angle)
                inputs[wrong, 2:] *= np.exp(rng.uniform(-0.2, 0.2, (n, 1)))

        return inputs, observed


class DetectorReplay:
    """
    A entrada e a caixa do PROPRIO detector (attach_detections): o erro, as
    falhas e a correlacao temporal do erro sao os reais. Por cima, blocos de
    ausencia simulados (as falhas reais sao quase todas curtas; o estado
    precisa aprender a rodar sem observacao por mais tempo).

    Args:
        occlusion_prob, occlusion_len: como em DetectionNoise.
        fallback: DetectionNoise usado so no primeiro passo da janela quando
            o detector nao achou a pessoa ali (a track nasce de uma deteccao).
    """

    def __init__(self, occlusion_prob=0.3, occlusion_len=(1, 40), fallback=None):
        self.occlusion_prob = occlusion_prob
        self.occlusion_len = occlusion_len
        self.fallback = fallback if fallback is not None else DetectionNoise()

    def sample(self, boxes, visibility, rng, detections=None):

        if detections is None:
            raise ValueError("DetectorReplay precisa de traj['detections'] (attach_detections)")

        L = len(boxes)
        observed = ~np.isnan(detections).any(axis=1)
        inputs = np.where(observed[:, None], detections, boxes)

        if not observed[0]:
            first, _ = self.fallback.sample(boxes[:1], visibility[:1], rng)
            inputs[0] = first[0]
            observed[0] = True

        if self.occlusion_prob > 0 and L > 2 and rng.random() < self.occlusion_prob:
            lo, hi = self.occlusion_len
            n = int(rng.integers(lo, hi + 1))
            s = int(rng.integers(1, max(2, L - 1)))
            observed[s:s + n] = False

        return inputs, observed


class TrajectoryDataset(Dataset):
    """
    Janelas de `length` passos das trajetorias, no formato do MotionModel:

        inputs    (T, 4)  caixas "detectadas" (so valem onde observed = 1)
        boxes     (T, 4)  caixas verdadeiras (o alvo)
        observed  (T, 1)  1 = houve deteccao no passo
        dt        (T, 1)  quadros desde o passo anterior, em 1/30 s
        valid     (T,)    1 nos passos reais, 0 no preenchimento
        visibility (T,)

    Args:
        trajectories: lista de extract_trajectories (varias sequencias).
        length, stride: janela e passo entre janelas.
        noise: DetectionNoise, DetectorReplay, ou None (entrada = GT,
            tudo observado).
        fixed: sorteia o ruido UMA vez por janela e guarda (avaliacao
            deterministica); senao sorteia de novo a cada epoca.
        flip_prob: probabilidade de espelhar a janela na horizontal (o
            movimento em x troca de sinal; so diferencas de posicao
            entram no modelo, entao a posicao absoluta nao importa).
        reverse_prob: probabilidade de inverter o tempo da janela
            (pedestres e cameras andando "para tras" continuam plausiveis).
        seed: semente.
    """

    def __init__(self, trajectories, length=32, stride=16, noise=None, fixed=False,
                 flip_prob=0.0, reverse_prob=0.0, seed=0):

        self.trajectories = trajectories
        self.length = length
        self.noise = noise
        self.fixed = fixed
        self.flip_prob = flip_prob
        self.reverse_prob = reverse_prob
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache = {}

        self.windows = []

        for k, traj in enumerate(trajectories):

            L = len(traj["frames"])
            starts = list(range(0, max(L - length, 0) + 1, stride))

            if starts[-1] + length < L:
                starts.append(L - length)

            self.windows.extend((k, s) for s in starts)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):

        if self.fixed and index in self.cache:
            return self.cache[index]

        k, start = self.windows[index]
        traj = self.trajectories[k]

        sl = slice(start, start + self.length)
        frames = traj["frames"][sl]
        boxes = traj["boxes"][sl]
        visibility = traj["visibility"][sl]
        detections = traj["detections"][sl] if "detections" in traj else None

        L, T = len(frames), self.length

        rng = np.random.default_rng((self.seed, index)) if self.fixed else self.rng

        # augmentation geometrica / temporal (so no treino: fixed=False)
        if not self.fixed:
            if self.flip_prob > 0 and rng.random() < self.flip_prob:
                boxes = boxes.copy()
                boxes[:, 0] = -boxes[:, 0]
                if detections is not None:
                    detections = detections.copy()
                    detections[:, 0] = -detections[:, 0]
            if self.reverse_prob > 0 and rng.random() < self.reverse_prob:
                boxes, visibility = boxes[::-1].copy(), visibility[::-1].copy()
                frames = frames[-1] - frames[::-1]
                if detections is not None:
                    detections = detections[::-1].copy()

        if self.noise is None:
            inputs, observed = boxes.copy(), np.ones(L, dtype=bool)
        else:
            inputs, observed = self.noise.sample(boxes, visibility, rng, detections=detections)

        dt = np.full(L, traj["dt_unit"])
        dt[1:] = np.diff(frames) * traj["dt_unit"]

        def pad(a):
            a = np.asarray(a, dtype=np.float32)
            if L == T:
                return a
            fill = np.repeat(a[-1:], T - L, axis=0)     # repete o ultimo passo (nada de zeros no log)
            return np.concatenate([a, fill], axis=0)

        observed_padded = np.zeros(T, dtype=np.float32)
        observed_padded[:L] = observed

        valid = np.zeros(T, dtype=np.float32)
        valid[:L] = 1.0

        item = {
            "inputs": torch.from_numpy(pad(inputs)),
            "boxes": torch.from_numpy(pad(boxes)),
            "observed": torch.from_numpy(observed_padded[:, None]),
            "dt": torch.from_numpy(pad(dt)[:, None]),
            "valid": torch.from_numpy(valid),
            "visibility": torch.from_numpy(pad(visibility)),
        }

        if self.fixed:
            self.cache[index] = item

        return item
