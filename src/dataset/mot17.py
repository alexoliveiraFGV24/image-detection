"""
MOT17 -- leitura das anotacoes, das deteccoes publicas e dos quadros.

Estrutura esperada em `root` (padrao "../data", relativo a reports/):

    MOT17Labels/train/MOT17-02-FRCNN/{gt/gt.txt, det/det.txt, seqinfo.ini}
    MOT17Labels/train/MOT17-02-DPM/det/det.txt          (idem SDP)
    MOT17Det/train/MOT17-02/img1/000001.jpg              (quadros, opcional)

Os quadros tambem sao procurados em MOT17/train/MOT17-02-<DET>/img1 (o
pacote completo, que repete as imagens tres vezes) -- o que existir.

Tudo sai no formato de TABELA do repositorio (src/nn/boxes.py):
(frame, id, x1, y1, x2, y2, conf), frames 0-indexados.

Ground truth: so as linhas que o MOTChallenge avalia -- conf = 1 (a
pessoa conta) e classe 1 (pedestre); a coluna `conf` da tabela recebe a
VISIBILIDADE. As demais classes ficam em `gt_all` para o pre-processamento
oficial dos distratores (`remove_distractor_matches`).
"""

import os
import configparser

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.nn.boxes import (
    FRAME, ID, X1, Y1, Y2, CONF,
    mot_to_table, box_iou_matrix, split_by_id, empty_table,
)


# ============================================================
# AS SEQUENCIAS
# ============================================================

DETECTORS = ("DPM", "FRCNN", "SDP")

# As 7 sequencias com ground truth (o "train" do MOTChallenge). As de
# teste nao tem GT publico, entao nao servem para avaliar.
SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

# Split POR SEQUENCIA (nunca por quadro). Justificativa no notebook da
# Parte 1: a validacao tem uma camera parada e uma em movimento, dia e
# noite; o treino fica com os extremos de densidade (04 e 05), a unica
# sequencia de baixa resolucao / 14 fps (05) e a camera de onibus (13).
SPLIT = {
    "train": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-11", "MOT17-13"],
    "val": ["MOT17-09", "MOT17-10"],
}

# Descricoes da pagina de dados do MOT17 (motchallenge.net/data/MOT17).
SEQUENCE_META = {
    "MOT17-02": {"camera": "parada", "viewpoint": "nivel dos olhos", "scene": "praca, dia"},
    "MOT17-04": {"camera": "parada", "viewpoint": "elevado", "scene": "rua de pedestres, noite"},
    "MOT17-05": {"camera": "movel", "viewpoint": "nivel dos olhos", "scene": "rua, plataforma movel"},
    "MOT17-09": {"camera": "parada", "viewpoint": "baixo", "scene": "rua de pedestres"},
    "MOT17-10": {"camera": "movel", "viewpoint": "nivel dos olhos", "scene": "rua, noite"},
    "MOT17-11": {"camera": "movel", "viewpoint": "nivel dos olhos", "scene": "shopping, camera andando"},
    "MOT17-13": {"camera": "movel", "viewpoint": "elevado", "scene": "cruzamento, de dentro de um onibus"},
}

PEDESTRIAN = 1
# Classes que o MOTChallenge trata como "nao e erro detectar":
# pessoa em veiculo, pessoa parada/estatica, distrator, reflexo.
DISTRACTOR_CLASSES = (2, 7, 8, 12)


def read_seqinfo(path):
    """seqinfo.ini -> dict (name, frameRate, seqLength, imWidth, imHeight, ...)."""

    parser = configparser.ConfigParser()
    parser.read(path)
    info = dict(parser["Sequence"])

    for key in ("frameRate", "seqLength", "imWidth", "imHeight"):
        key_lower = key.lower()
        if key_lower in info:
            info[key] = int(info.pop(key_lower))

    return info


def _load_txt(path):
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    return rows if rows.size else np.zeros((0, 10))


# ============================================================
# UMA SEQUENCIA
# ============================================================


class MOT17Sequence:
    """
    Uma sequencia do MOT17 (as anotacoes sao as mesmas nas tres pastas
    -DPM/-FRCNN/-SDP; so muda det.txt).

    Attributes:
        name: "MOT17-02".
        info: seqinfo (frameRate, seqLength, imWidth, imHeight).
        fps, n_frames, width, height.
        gt: tabela dos pedestres avaliados (conf = visibilidade).
        gt_all: linhas cruas do gt.txt (todas as classes, 1-indexadas).
        meta: camera / ponto de vista / cena.
    """

    def __init__(self, name, root="../data"):

        self.name = name
        self.root = root
        self.labels_dir = os.path.join(root, "MOT17Labels", "train")

        base = os.path.join(self.labels_dir, f"{name}-FRCNN")

        if not os.path.isdir(base):
            raise FileNotFoundError(f"{base} nao encontrado -- baixe o MOT17Labels (README, secao 2)")

        self.info = read_seqinfo(os.path.join(base, "seqinfo.ini"))
        self.fps = self.info["frameRate"]
        self.n_frames = self.info["seqLength"]
        self.width = self.info["imWidth"]
        self.height = self.info["imHeight"]

        self.gt_all = _load_txt(os.path.join(base, "gt", "gt.txt"))

        keep = (self.gt_all[:, 6] == 1) & (self.gt_all[:, 7] == PEDESTRIAN)
        self.gt = mot_to_table(self.gt_all[keep], conf_column=8)

        self.meta = SEQUENCE_META.get(name, {})

        self._image_dir = None

    def __repr__(self):
        return (f"MOT17Sequence({self.name}, {self.n_frames} quadros @ {self.fps} fps, "
                f"{self.width}x{self.height}, {len(np.unique(self.gt[:, ID]))} ids)")

    @property
    def image_size(self):
        return (self.width, self.height)

    # --------------------------------------------------------
    # deteccoes publicas
    # --------------------------------------------------------

    def public_detections(self, detector="SDP", min_score=None):
        """
        det.txt do detector publico -> tabela (frame, -1, x1, y1, x2, y2, score).

        Args:
            detector: "DPM" | "FRCNN" | "SDP".
            min_score: descarta deteccoes com score < min_score (None = todas).
        """

        path = os.path.join(self.labels_dir, f"{self.name}-{detector}", "det", "det.txt")
        det = mot_to_table(_load_txt(path), conf_column=6)
        det[:, ID] = -1

        if min_score is not None:
            det = det[det[:, CONF] >= min_score]

        return det

    # --------------------------------------------------------
    # quadros
    # --------------------------------------------------------

    def _find_image_dir(self):

        candidates = [os.path.join(self.root, "MOT17Det", "train", self.name, "img1")]
        candidates += [os.path.join(self.root, "MOT17", "train", f"{self.name}-{d}", "img1") for d in DETECTORS]
        candidates += [os.path.join(self.root, "MOT17", "train", self.name, "img1")]

        for c in candidates:
            if os.path.isdir(c):
                return c

        return None

    @property
    def image_dir(self):
        if self._image_dir is None:
            self._image_dir = self._find_image_dir()
        return self._image_dir

    @property
    def has_images(self):
        return self.image_dir is not None

    def image_path(self, t):
        """Caminho do quadro t (0-indexado)."""

        if not self.has_images:
            raise FileNotFoundError(f"quadros de {self.name} nao encontrados em {self.root} (README, secao 2)")

        return os.path.join(self.image_dir, f"{t + 1:06d}{self.info.get('imext', '.jpg')}")

    def image(self, t):
        """Quadro t como np.ndarray RGB uint8 (H, W, 3)."""

        from PIL import Image
        return np.asarray(Image.open(self.image_path(t)).convert("RGB"))


def load_sequences(names=None, root="../data"):
    """dict nome -> MOT17Sequence, na ordem dada."""

    names = SEQUENCES if names is None else names
    return {name: MOT17Sequence(name, root) for name in names}


# ============================================================
# PRE-PROCESSAMENTO DOS DISTRATORES (regra oficial do MOTChallenge)
# ============================================================


def remove_distractor_matches(pred, gt_all, iou_threshold=0.5):
    """
    Remove das previsoes as caixas que cobrem DISTRATORES: pessoas em
    veiculos, pessoas estaticas, reflexos, etc. (classes 2, 7, 8, 12).
    O detector nao errou ao achar essas pessoas, entao elas nao podem
    contar como falso positivo -- mas tambem nao entram no GT avaliado.

    Regra do avaliador oficial: por quadro, Hungarian entre as previsoes
    e TODO o GT com conf = 1 ou classe distratora (IoU >= limiar); as
    previsoes casadas com um distrator saem.

    Args:
        pred: tabela (N, 7).
        gt_all: linhas cruas do gt.txt (MOT17Sequence.gt_all).

    Returns:
        tabela filtrada.
    """

    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)

    if len(pred) == 0:
        return pred

    gt_all = np.asarray(gt_all, dtype=np.float64)

    relevant = ((gt_all[:, 6] == 1) & (gt_all[:, 7] == PEDESTRIAN)) | np.isin(gt_all[:, 7], DISTRACTOR_CLASSES)
    gt_rel = gt_all[relevant]

    if len(gt_rel) == 0:
        return pred

    gt_table = mot_to_table(gt_rel)
    is_distractor = np.isin(gt_rel[:, 7], DISTRACTOR_CLASSES)

    gt_frames = gt_table[:, FRAME].astype(int)
    order = np.argsort(gt_frames, kind="stable")
    gt_frames_sorted = gt_frames[order]

    keep = np.ones(len(pred), dtype=bool)

    pred_frames = pred[:, FRAME].astype(int)

    for frame in np.unique(pred_frames):

        lo, hi = np.searchsorted(gt_frames_sorted, [frame, frame + 1])

        if lo == hi:
            continue

        g_idx = order[lo:hi]

        if not is_distractor[g_idx].any():
            continue

        p_idx = np.flatnonzero(pred_frames == frame)

        iou = box_iou_matrix(pred[p_idx, X1:Y2 + 1], gt_table[g_idx, X1:Y2 + 1])
        cost = np.where(iou >= iou_threshold, 1.0 - iou, 1e6)
        rows, cols = linear_sum_assignment(cost)

        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_threshold and is_distractor[g_idx[c]]:
                keep[p_idx[r]] = False

    return pred[keep]


def evaluate_sequence(pred, seq, iou_threshold=0.5, detection_map=True, **kwargs):
    """
    Avalia uma tabela prevista contra o GT de uma sequencia do MOT17, com
    o pre-processamento dos distratores. Devolve o dict de
    src.nn.metrics.evaluate_tracking.
    """

    from src.nn.metrics import evaluate_tracking

    pred = remove_distractor_matches(pred, seq.gt_all)

    return evaluate_tracking(pred, seq.gt, iou_threshold=iou_threshold, detection_map=detection_map, **kwargs)


# ============================================================
# EIXOS DE DIFICULDADE (medidos no GT)
# ============================================================


def _runs(mask):
    """Comprimentos das sequencias de True consecutivos em `mask`."""

    if mask.size == 0:
        return np.zeros(0, dtype=int)

    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    return ends - starts


def occlusion_durations(gt, max_visibility=0.25, only_interior=True):
    """
    Duracoes (em quadros) das oclusoes do GT: por identidade, sequencias
    de quadros consecutivos com visibility < max_visibility. O MOT17
    anota a pessoa inteira mesmo escondida -- e esse "ouro" que permite
    medir quanto tempo cada uma fica atras de algo.

    Args:
        only_interior: so conta oclusoes com a pessoa visivel antes E
            depois (some e volta) -- as que exigem memoria.
    """

    durations = []

    for _, rows in split_by_id(gt).items():

        frames = rows[:, FRAME].astype(int)
        vis = rows[:, CONF]

        # quadros sem anotacao no meio da track contam como ocluidos
        full = np.full(frames.max() - frames.min() + 1, 0.0)
        full[frames - frames.min()] = vis
        present = np.zeros_like(full, dtype=bool)
        present[frames - frames.min()] = True

        hidden = (full < max_visibility) | ~present

        if only_interior:
            padded = np.concatenate([[False], hidden, [False]])
            diff = np.diff(padded.astype(int))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)

            for s, e in zip(starts, ends):
                if s > 0 and e < len(hidden):
                    durations.append(e - s)
        else:
            durations.extend(_runs(hidden).tolist())

    return np.asarray(durations, dtype=int)


def consecutive_iou(gt, step=1):
    """
    IoU entre a caixa de cada identidade no quadro t e no quadro t + step.
    E exatamente o que a associacao ingenua usa: se esse IoU cai abaixo
    do limiar, nem um detector perfeito salva a identidade.
    """

    values = []

    for _, rows in split_by_id(gt).items():

        frames = rows[:, FRAME].astype(int)
        boxes = rows[:, X1:Y2 + 1]

        index = {f: k for k, f in enumerate(frames)}

        for k, f in enumerate(frames):
            j = index.get(f + step)
            if j is not None:
                a, b = boxes[k], boxes[j]
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                values.append(inter / union if union > 0 else 0.0)

    return np.asarray(values)


def sequence_statistics(seq, occlusion_visibility=0.25, iou_threshold=0.5):
    """
    Os candidatos a eixo de dificuldade, todos medidos no GT:

        density           pedestres por quadro
        ids               identidades unicas
        frac_occluded     fracao das caixas com visibilidade < 0.5
        occ_mean_s / occ_p90_s  duracao media / p90 das oclusoes (em
                          SEGUNDOS -- as sequencias tem fps diferentes)
        occ_mean_frames   idem, em quadros
        consec_iou        IoU medio da mesma pessoa entre quadros seguidos
        frac_low_iou      fracao dos pares consecutivos com IoU < limiar
        box_height        altura mediana das caixas (px)
        camera            parada / movel
    """

    gt = seq.gt

    occ = occlusion_durations(gt, max_visibility=occlusion_visibility)
    iou = consecutive_iou(gt)

    return {
        "sequence": seq.name,
        "fps": seq.fps,
        "frames": seq.n_frames,
        "resolution": f"{seq.width}x{seq.height}",
        "camera": seq.meta.get("camera", "?"),
        "viewpoint": seq.meta.get("viewpoint", "?"),
        "ids": int(len(np.unique(gt[:, ID]))),
        "density": float(len(gt) / seq.n_frames),
        "frac_occluded": float(np.mean(gt[:, CONF] < 0.5)),
        "n_occlusions": int(len(occ)),
        "occ_mean_frames": float(occ.mean()) if len(occ) else 0.0,
        "occ_mean_s": float(occ.mean() / seq.fps) if len(occ) else 0.0,
        "occ_p90_s": float(np.percentile(occ, 90) / seq.fps) if len(occ) else 0.0,
        "consec_iou": float(iou.mean()),
        "frac_low_iou": float(np.mean(iou < iou_threshold)),
        "box_height": float(np.median(gt[:, Y2] - gt[:, Y1])),
    }


def detections_to_mot(table, path):
    """Grava uma tabela de deteccoes/trajetorias no formato do MOTChallenge."""

    from src.nn.boxes import table_to_mot

    os.makedirs(os.path.dirname(path), exist_ok=True)
    table_to_mot(table, path)


def load_mot_file(path):
    """Le um .txt no formato do MOTChallenge (como o gravado acima)."""

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return empty_table()

    return mot_to_table(_load_txt(path), conf_column=6)
