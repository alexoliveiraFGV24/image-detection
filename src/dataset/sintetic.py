"""
Parte 0 -- o ambiente controlado.

1. ``generate_video``    videos 128x128 de 30-60 quadros com 5-15 elipses
                         em movimento, tamanhos variados, ruido e contraste
                         variaveis. Parametros expostos: numero de objetos,
                         velocidade tipica e DURACAO DA OCLUSAO.

   As elipses sao desenhadas em ORDEM DE PROFUNDIDADE (da mais distante
   para a mais proxima, a proxima sobrescreve), entao uma passa atras da
   outra e realmente desaparece. A visibilidade de cada objeto em cada
   quadro e medida no proprio mapa de rotulos renderizado -- e a coluna
   ``visibility`` do gt.txt do MOT17.

2. ``simulate_detector``  recebe as caixas verdadeiras e as estraga de
                         proposito: descarta p% delas, adiciona ruido nas
                         coordenadas, injeta falsos positivos. Opera sobre
                         tabelas (N, 7), entao serve igual para o MOT17
                         (Parte 5).

3. ``SyntheticVideoDataset``  varios videos, gerados sob demanda a partir
                         de uma seed base (determinismo sem ocupar memoria).

Ground truth: tabela (N, 7) = (frame, id, x1, y1, x2, y2, visibility),
com a caixa da elipse INTEIRA (como o MOT17 anota a pessoa inteira mesmo
ocluida); ``video.visible_boxes`` traz, alinhada linha a linha, a caixa
so da parte visivel (NaN se totalmente ocluida).
"""

import numpy as np

from src.nn.boxes import make_table, empty_table, clip_boxes, FRAME, ID, X1, Y2, CONF


# ============================================================
# RENDERIZACAO
# ============================================================


def _ellipse_mask(size, cx, cy, rx, ry):
    """Mascara booleana (size, size) de uma elipse alinhada aos eixos."""

    ys, xs = np.mgrid[0:size, 0:size]
    return ((xs + 0.5 - cx) / rx) ** 2 + ((ys + 0.5 - cy) / ry) ** 2 <= 1.0


def _mask_box(mask):
    """Caixa xyxy (em pixels) dos pixels verdadeiros; None se vazia."""

    ys, xs = np.nonzero(mask)

    if xs.size == 0:
        return None

    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


# ============================================================
# O VIDEO
# ============================================================


class SyntheticVideo:
    """
    Resultado de ``generate_video``.

    Attributes:
        frames: (T, H, W) float32 em [0, 1], escala de cinza.
        label_maps: (T, H, W) int16, 0 = fundo, id = objeto VISIVEL ali.
        gt: tabela (N, 7) (frame, id, x1, y1, x2, y2, visibility) com a
            caixa inteira do objeto (clipada a imagem).
        visible_boxes: (N, 4) caixa da parte visivel (NaN se ocluida),
            alinhada com `gt`.
        objects: dict id -> {rx, ry, intensity, depth, speed}.
        occlusions: lista de oclusoes PLANEJADAS
            {occludee, occluder, t_mid, target_duration}.
        params: os parametros usados na geracao.
    """

    def __init__(self, frames, label_maps, gt, visible_boxes, objects, occlusions, params):
        self.frames = frames
        self.label_maps = label_maps
        self.gt = gt
        self.visible_boxes = visible_boxes
        self.objects = objects
        self.occlusions = occlusions
        self.params = params

    @property
    def n_frames(self):
        return self.frames.shape[0]

    @property
    def size(self):
        return self.frames.shape[1]

    def to_tensor(self):
        """(T, 3, H, W) float32 -- o cinza replicado em 3 canais."""

        import torch
        return torch.from_numpy(np.repeat(self.frames[:, None], 3, axis=1)).float()

    def visibility_of(self, track_id):
        """(T,) visibilidade do objeto em cada quadro (0 onde ausente)."""

        vis = np.zeros(self.n_frames, dtype=np.float64)
        rows = self.gt[self.gt[:, ID] == track_id]
        vis[rows[:, FRAME].astype(int)] = rows[:, CONF]
        return vis

    def trajectory_of(self, track_id):
        """(T, 4) caixa inteira do objeto por quadro (NaN onde ausente)."""

        traj = np.full((self.n_frames, 4), np.nan)
        rows = self.gt[self.gt[:, ID] == track_id]
        traj[rows[:, FRAME].astype(int)] = rows[:, X1:Y2 + 1]
        return traj

    def occlusion_intervals(self, max_visibility=0.0):
        """
        Intervalos MEDIDOS de oclusao: por id, lista de (inicio, fim)
        inclusivos em que visibility <= max_visibility, com o objeto
        visivel antes e depois. E a verificacao do requisito "some por
        N quadros e volta".
        """

        intervals = {}

        for track_id in np.unique(self.gt[:, ID]).astype(int):

            vis = self.visibility_of(track_id)
            hidden = vis <= max_visibility

            runs = []
            t = 0

            while t < self.n_frames:

                if not hidden[t]:
                    t += 1
                    continue

                start = t
                while t < self.n_frames and hidden[t]:
                    t += 1
                end = t - 1

                # so conta se some E volta
                if start > 0 and end < self.n_frames - 1:
                    runs.append((start, end))

            if runs:
                intervals[track_id] = runs

        return intervals


# ============================================================
# O GERADOR
# ============================================================


def generate_video(
    n_objects=8,
    speed=2.0,
    occlusion_duration=0,
    n_occlusions=None,
    approach_speed=None,
    n_frames=None,
    size=128,
    radius_range=(4, 14),
    noise_std=None,
    contrast=None,
    brightness=None,
    seed=None,
):
    """
    Gera um video sintetico de elipses em movimento.

    Args:
        n_objects: numero de elipses (o enunciado pede 5 a 15).
        speed: velocidade tipica em px/quadro; cada objeto sorteia a
            sua em U(0.5, 1.5) * speed, com direcao aleatoria; os
            objetos ricocheteiam nas bordas.
        occlusion_duration: duracao ALVO (em quadros) de cada oclusao
            planejada. 0 = nenhuma oclusao planejada (as elipses ainda
            podem se cruzar por acaso).
        n_occlusions: quantas oclusoes planejar (padrao: 1 se
            occlusion_duration > 0). Cada uma usa um par distinto.
        approach_speed: velocidade RELATIVA (px/quadro) com que o
            ocultador alcanca e depois deixa o ocultado (padrao:
            max(1.5, speed)). Controla a duracao da oclusao parcial.
        n_frames: duracao (padrao: sorteia em [30, 60]).
        size: lado da imagem.
        radius_range: (min, max) dos semi-eixos das elipses.
        noise_std: desvio do ruido gaussiano (padrao: U(0, 0.10)).
        contrast, brightness: (padrao: U(0.7, 1.4) e U(-0.1, 0.1)).
        seed: semente (determinismo).

    Como a oclusao planejada funciona: o ocultado A anda com a sua
    velocidade; o ocultador B (maior que A em `margin` px nos dois
    eixos, e mais PROXIMO da camera) vem pela mesma linha com velocidade
    relativa `approach_speed`, alcanca A, anda JUNTO com A (deslocamento
    relativo zero) por `occlusion_duration` quadros centrados em t_mid,
    e depois se afasta com a mesma velocidade relativa. Enquanto andam
    juntos B cobre A por inteiro; as fases de aproximacao/afastamento
    (oclusao parcial) duram so (rx_A + rx_B) / approach_speed quadros.
    A duracao REAL e medida depois no mapa de rotulos
    (``occlusion_intervals``) -- e o que o notebook verifica.

    Returns:
        SyntheticVideo
    """

    rng = np.random.default_rng(seed)

    if n_frames is None:
        n_frames = int(rng.integers(30, 61))

    if n_occlusions is None:
        n_occlusions = 1 if occlusion_duration > 0 else 0

    if occlusion_duration <= 0:
        n_occlusions = 0

    if approach_speed is None:
        approach_speed = max(1.5, float(speed))

    if 2 * n_occlusions > n_objects:
        raise ValueError("cada oclusao planejada precisa de um par distinto de objetos")

    if noise_std is None:
        noise_std = float(rng.uniform(0.0, 0.10))

    if contrast is None:
        contrast = float(rng.uniform(0.7, 1.4))

    if brightness is None:
        brightness = float(rng.uniform(-0.1, 0.1))

    r_min, r_max = radius_range
    T = n_frames

    # --------------------------------------------------------
    # objetos: raio, intensidade, profundidade, velocidade
    # --------------------------------------------------------

    rx = rng.uniform(r_min, r_max, n_objects)
    ry = rng.uniform(r_min, r_max, n_objects)
    intensity = rng.uniform(0.55, 1.0, n_objects)
    depth = rng.permutation(n_objects).astype(np.float64)

    obj_speed = rng.uniform(0.5, 1.5, n_objects) * speed
    angle = rng.uniform(0, 2 * np.pi, n_objects)
    vel = np.stack([np.cos(angle), np.sin(angle)], axis=1) * obj_speed[:, None]

    centers = np.zeros((T, n_objects, 2))

    # --------------------------------------------------------
    # oclusoes planejadas: pares (A = ocultado, B = ocultador)
    # --------------------------------------------------------

    occlusions = []
    driven = {}         # B -> (A, t_mid, u, margin)

    for k in range(n_occlusions):

        a, b = 2 * k, 2 * k + 1
        margin = float(rng.uniform(3.0, 5.0))

        # A cabe dentro de B com folga `margin` nos dois eixos
        rx[a] = min(rx[a], r_max - margin)
        ry[a] = min(ry[a], r_max - margin)
        rx[b] = rx[a] + margin
        ry[b] = ry[a] + margin

        # B mais proximo da camera que A
        if depth[b] < depth[a]:
            depth[a], depth[b] = depth[b], depth[a]

        half = occlusion_duration / 2.0
        t_mid = float(rng.uniform(half + 3, max(half + 3, T - 1 - half - 3)))

        if np.linalg.norm(vel[a]) > 1e-6:
            u = vel[a] / np.linalg.norm(vel[a])
        else:
            theta = rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(theta), np.sin(theta)])

        driven[b] = (a, t_mid, u, margin)

        occlusions.append({
            "occludee": a + 1,
            "occluder": b + 1,
            "t_mid": t_mid,
            "target_duration": int(occlusion_duration),
            "margin": margin,
        })

    # --------------------------------------------------------
    # trajetorias: movimento linear com ricochete nas bordas
    # --------------------------------------------------------

    pos = np.zeros((n_objects, 2))

    for i in range(n_objects):
        pos[i, 0] = rng.uniform(rx[i], size - rx[i])
        pos[i, 1] = rng.uniform(ry[i], size - ry[i])

    v = vel.copy()

    for t in range(T):

        centers[t] = pos

        pos = pos + v

        # ricochete: reflete a velocidade e devolve para dentro
        for i in range(n_objects):

            if i in driven:
                continue

            for axis, r in ((0, rx[i]), (1, ry[i])):

                if pos[i, axis] < r:
                    pos[i, axis] = 2 * r - pos[i, axis]
                    v[i, axis] = -v[i, axis]

                elif pos[i, axis] > size - r:
                    pos[i, axis] = 2 * (size - r) - pos[i, axis]
                    v[i, axis] = -v[i, axis]

    # os ocultadores seguem o ocultado + deslocamento relativo:
    # zero durante a janela de oclusao, linear fora dela. B ainda cobre
    # A enquanto |deslocamento| <= margin, entao a janela de parada e
    # encurtada em 2 * margin / approach_speed para a oclusao total
    # medida ficar ~ occlusion_duration.
    for b, (a, t_mid, u, margin) in driven.items():

        half = max(0.0, occlusion_duration / 2.0 - margin / approach_speed)

        rel = np.arange(T) - t_mid
        offset = np.sign(rel) * np.clip(np.abs(rel) - half, 0.0, None) * approach_speed
        centers[:, b] = centers[:, a] + offset[:, None] * u[None, :]

    # --------------------------------------------------------
    # renderizacao em ordem de profundidade + ground truth
    # --------------------------------------------------------

    background = float(rng.uniform(0.08, 0.30))

    frames = np.zeros((T, size, size), dtype=np.float32)
    label_maps = np.zeros((T, size, size), dtype=np.int16)

    gt_rows = []
    visible_rows = []

    order = np.argsort(depth)          # distante -> proximo

    for t in range(T):

        image = np.full((size, size), background, dtype=np.float32)
        labels = np.zeros((size, size), dtype=np.int16)

        masks = {}

        for i in order:

            cx, cy = centers[t, i]
            mask = _ellipse_mask(size, cx, cy, rx[i], ry[i])

            if not mask.any():
                continue

            masks[i] = mask
            image[mask] = intensity[i]
            labels[mask] = i + 1

        for i, mask in masks.items():

            full_box = _mask_box(mask)
            visible_mask = labels == (i + 1)
            visibility = visible_mask.sum() / mask.sum()

            gt_rows.append((t, i + 1, *full_box, visibility))

            vbox = _mask_box(visible_mask)
            visible_rows.append(vbox if vbox is not None else np.full(4, np.nan))

        # contraste, brilho, ruido (por quadro)
        image = image * contrast + brightness
        image = image + rng.normal(0.0, noise_std, image.shape).astype(np.float32)

        frames[t] = np.clip(image, 0.0, 1.0)
        label_maps[t] = labels

    gt = np.asarray(gt_rows, dtype=np.float64).reshape(-1, 7)
    visible_boxes = np.asarray(visible_rows, dtype=np.float64).reshape(-1, 4)

    objects = {
        i + 1: {
            "rx": float(rx[i]), "ry": float(ry[i]),
            "intensity": float(intensity[i]),
            "depth": float(depth[i]),
            "speed": float(np.linalg.norm(vel[i])),
        }
        for i in range(n_objects)
    }

    params = {
        "n_objects": n_objects, "speed": speed,
        "occlusion_duration": occlusion_duration, "n_occlusions": n_occlusions,
        "approach_speed": approach_speed, "n_frames": T, "size": size, "radius_range": tuple(radius_range),
        "noise_std": noise_std, "contrast": contrast, "brightness": brightness,
        "seed": seed,
    }

    return SyntheticVideo(frames, label_maps, gt, visible_boxes, objects, occlusions, params)


# ============================================================
# O SIMULADOR DE DETECTOR
# ============================================================


def simulate_detector(
    gt,
    visible_boxes=None,
    drop_rate=0.0,
    noise_std=0.0,
    fp_rate=0.0,
    min_visibility=0.5,
    image_size=128,
    fp_size_range=(8, 28),
    seed=None,
):
    """
    Estraga as caixas verdadeiras de proposito.

    Args:
        gt: tabela (N, 7) com `conf` = visibilidade.
        visible_boxes: (N, 4) caixa da parte visivel (do gerador). Se
            dada, a deteccao "ideal" de um objeto parcialmente ocluido
            e essa caixa, nao a inteira -- mais realista, mas
            desalinha a deteccao do GT (que e a caixa inteira, como no
            MOT17) e penaliza o IoU mesmo com rastreamento perfeito.
            Padrao None: caixa inteira, e a oclusao vira SO falta de
            deteccao (via `min_visibility`).
        drop_rate: fracao p das deteccoes descartadas ao acaso.
        noise_std: desvio do ruido gaussiano nas coordenadas, RELATIVO
            ao tamanho da caixa (0.1 = 10% da largura/altura).
        fp_rate: numero medio de falsos positivos por quadro (Poisson).
        min_visibility: objetos com visibilidade abaixo disto nao sao
            detectados -- a oclusao vira falta de deteccao.
        image_size: lado (ou (w, h)) da imagem, para os falsos positivos
            e para clipar as caixas.
        fp_size_range: (min, max) do lado dos falsos positivos.
        seed: semente.

    Returns:
        tabela (M, 7) de deteccoes (frame, -1, x1, y1, x2, y2, score).
        O score cresce com a visibilidade; falsos positivos recebem
        scores baixos-medios (U(0.1, 0.6)).
    """

    rng = np.random.default_rng(seed)

    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 7)

    if isinstance(image_size, (int, float)):
        width = height = float(image_size)
    else:
        width, height = map(float, image_size)

    if gt.shape[0] == 0:
        frames_all = []
    else:
        frames_all = np.unique(gt[:, FRAME]).astype(int)

    boxes = gt[:, X1:Y2 + 1].copy()

    if visible_boxes is not None:
        visible_boxes = np.asarray(visible_boxes, dtype=np.float64).reshape(-1, 4)
        has_visible = ~np.isnan(visible_boxes).any(axis=1)
        boxes[has_visible] = visible_boxes[has_visible]

    visibility = gt[:, CONF]

    # 1. quem e detectavel: visivel o bastante E nao descartado
    detectable = visibility >= min_visibility
    detectable &= rng.uniform(size=len(gt)) >= drop_rate

    frames = gt[detectable, FRAME]
    boxes = boxes[detectable]
    visibility = visibility[detectable]

    # 2. ruido nas coordenadas (relativo ao tamanho)
    if noise_std > 0 and len(boxes):

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        scale = np.stack([w, h, w, h], axis=1)

        boxes = boxes + rng.normal(0.0, 1.0, boxes.shape) * noise_std * scale

        # garante x2 > x1, y2 > y1
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1.0)

    scores = np.clip(visibility - np.abs(rng.normal(0.0, 0.05, len(boxes))), 0.05, 1.0)

    rows = [make_table(frames, -np.ones(len(frames)), boxes, scores)]

    # 3. falsos positivos
    if fp_rate > 0:

        fp_frames = []
        fp_boxes = []

        for t in frames_all:

            n_fp = rng.poisson(fp_rate)

            for _ in range(n_fp):

                w = rng.uniform(*fp_size_range)
                h = rng.uniform(*fp_size_range)
                cx = rng.uniform(0, width)
                cy = rng.uniform(0, height)

                fp_frames.append(t)
                fp_boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        if fp_frames:
            fp_scores = rng.uniform(0.1, 0.6, len(fp_frames))
            rows.append(make_table(fp_frames, -np.ones(len(fp_frames)), fp_boxes, fp_scores))

    det = np.concatenate(rows, axis=0) if rows else empty_table()

    if len(det):
        det[:, X1:Y2 + 1] = clip_boxes(det[:, X1:Y2 + 1], width, height)
        det = det[np.argsort(det[:, FRAME], kind="stable")]

    return det


# ============================================================
# VARIOS VIDEOS
# ============================================================


class SyntheticVideoDataset:
    """
    `n_videos` videos gerados sob demanda: o video i usa a seed
    `seed + i`, entao o dataset e deterministico sem guardar nada.

    Args:
        n_videos: quantos.
        seed: seed base.
        **params: repassados a `generate_video`.
    """

    def __init__(self, n_videos=32, seed=0, **params):
        self.n_videos = n_videos
        self.seed = seed
        self.params = params

    def __len__(self):
        return self.n_videos

    def __getitem__(self, index):

        if index < 0 or index >= self.n_videos:
            raise IndexError(index)

        return generate_video(seed=self.seed + index, **self.params)

    def __iter__(self):
        for i in range(self.n_videos):
            yield self[i]
