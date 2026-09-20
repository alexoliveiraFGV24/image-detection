"""
Utilidades de caixas (bounding boxes) e o NMS.

Convencao de caixas em todo o repositorio: ``(x1, y1, x2, y2)`` em pixels,
com x2 > x1 e y2 > y1 (canto superior esquerdo e canto inferior direito).

Convencao de TABELAS (ground truth, deteccoes e trajetorias previstas):
um ``np.ndarray`` float de shape ``(N, 7)`` com as colunas

    frame, id, x1, y1, x2, y2, conf

onde ``conf`` e a confianca da deteccao (deteccoes / previsoes) ou a
``visibility`` (ground truth). Deteccoes sem identidade usam ``id = -1``.
E a mesma informacao do ``gt.txt`` do MOT17 (frame, id, bb_left, bb_top,
bb_width, bb_height, conf, class, visibility), so que em xyxy e sem a
coluna de classe -- o MOT17 so tem pedestres.
"""

import numpy as np


FRAME, ID, X1, Y1, X2, Y2, CONF = range(7)
COLUMNS = ["frame", "id", "x1", "y1", "x2", "y2", "conf"]


# ============================================================
# CONVERSOES
# ============================================================


def xyxy_to_cxcywh(boxes):
    """(N, 4) xyxy -> (N, 4) centro + tamanho. Funciona com numpy e torch."""

    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    w = x2 - x1
    h = y2 - y1

    if isinstance(boxes, np.ndarray):
        return np.stack([x1 + w / 2, y1 + h / 2, w, h], axis=-1)

    import torch
    return torch.stack([x1 + w / 2, y1 + h / 2, w, h], dim=-1)


def cxcywh_to_xyxy(boxes):
    """(N, 4) centro + tamanho -> (N, 4) xyxy. Funciona com numpy e torch."""

    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]

    if isinstance(boxes, np.ndarray):
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    import torch
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def ltwh_to_xyxy(boxes):
    """Formato do MOT17 (left, top, width, height) -> xyxy."""

    boxes = np.asarray(boxes, dtype=np.float64)
    out = boxes.copy()
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def xyxy_to_ltwh(boxes):
    """xyxy -> formato do MOT17 (left, top, width, height)."""

    boxes = np.asarray(boxes, dtype=np.float64)
    out = boxes.copy()
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def clip_boxes(boxes, width, height):
    """Recorta as caixas ao retangulo [0, width] x [0, height]."""

    boxes = np.asarray(boxes, dtype=np.float64).copy()
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, width)
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, height)
    return boxes


def box_area(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    return (
        np.clip(boxes[..., 2] - boxes[..., 0], 0, None)
        * np.clip(boxes[..., 3] - boxes[..., 1], 0, None)
    )


# ============================================================
# IoU
# ============================================================


def box_iou_matrix(boxes_a, boxes_b):
    """
    Matriz de IoU (Na, Nb) entre dois conjuntos de caixas xyxy.

    Vetorizada: e a operacao mais chamada do repositorio (associacao a
    cada quadro, IDF1, ID switches, AP), entao nada de laco duplo.
    Caixas degeneradas (area zero) dao IoU 0.
    """

    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    # Intersecao: (Na, Nb)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    union = box_area(a)[:, None] + box_area(b)[None, :] - inter

    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def box_iou(box_a, box_b):
    """IoU entre duas caixas xyxy (escalar)."""

    return float(box_iou_matrix(np.asarray(box_a)[None], np.asarray(box_b)[None])[0, 0])


# ============================================================
# NMS  (o enunciado proibe torchvision.ops.nms)
# ============================================================


def nms(boxes, scores, iou_threshold=0.5, score_threshold=0.0):
    """
    Non-maximum suppression, como nos slides de deteccao (slide 35):

        1. descarta as caixas com score <= score_threshold;
        2. pega a caixa de maior score e a emite como predicao;
        3. descarta as demais com IoU >= iou_threshold com ela;
        4. repete enquanto sobrar caixa.

    Args:
        boxes: (N, 4) xyxy.
        scores: (N,).
        iou_threshold: limiar de supressao.
        score_threshold: caixas com score <= isto nem entram.

    Returns:
        indices (np.ndarray de int) das caixas mantidas, em ordem
        decrescente de score.
    """

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    candidates = np.flatnonzero(scores > score_threshold)

    if candidates.size == 0:
        return np.zeros(0, dtype=np.int64)

    # Ordena os candidatos por score decrescente
    order = candidates[np.argsort(-scores[candidates], kind="stable")]

    kept = []

    while order.size > 0:

        best = order[0]
        kept.append(int(best))

        if order.size == 1:
            break

        rest = order[1:]

        ious = box_iou_matrix(boxes[best][None], boxes[rest])[0]

        # Fica quem NAO se sobrepoe demais com a caixa emitida
        order = rest[ious < iou_threshold]

    return np.asarray(kept, dtype=np.int64)


# ============================================================
# TABELAS  (frame, id, x1, y1, x2, y2, conf)
# ============================================================


def make_table(frames, ids, boxes, conf):
    """Monta uma tabela (N, 7) a partir das colunas."""

    frames = np.asarray(frames, dtype=np.float64).reshape(-1)
    ids = np.asarray(ids, dtype=np.float64).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)

    return np.column_stack([frames, ids, boxes, conf])


def empty_table():
    return np.zeros((0, 7), dtype=np.float64)


def split_by_frame(table):
    """
    Dicionario frame -> linhas daquele quadro (ordenado por frame).

    O array devolvido por frame e uma VIEW ordenada da tabela (N_t, 7);
    para as caixas use ``rows[:, X1:Y2 + 1]``.
    """

    table = np.asarray(table, dtype=np.float64).reshape(-1, 7)

    if table.shape[0] == 0:
        return {}

    order = np.argsort(table[:, FRAME], kind="stable")
    table = table[order]

    frames, starts = np.unique(table[:, FRAME], return_index=True)
    ends = np.append(starts[1:], table.shape[0])

    return {
        int(f): table[s:e]
        for f, s, e in zip(frames, starts, ends)
    }


def split_by_id(table):
    """Dicionario id -> linhas daquela identidade, ordenadas por frame."""

    table = np.asarray(table, dtype=np.float64).reshape(-1, 7)

    if table.shape[0] == 0:
        return {}

    order = np.lexsort((table[:, FRAME], table[:, ID]))
    table = table[order]

    ids, starts = np.unique(table[:, ID], return_index=True)
    ends = np.append(starts[1:], table.shape[0])

    return {
        int(i): table[s:e]
        for i, s, e in zip(ids, starts, ends)
    }


def table_to_mot(table, path=None):
    """
    Converte para o formato de texto do MOTChallenge
    (frame, id, left, top, w, h, conf, -1, -1), 1-indexado no frame.
    Se `path` for dado, grava em disco.
    """

    table = np.asarray(table, dtype=np.float64).reshape(-1, 7)

    ltwh = xyxy_to_ltwh(table[:, X1:Y2 + 1])

    out = np.column_stack([
        table[:, FRAME] + 1,
        table[:, ID],
        ltwh,
        table[:, CONF],
        -np.ones(table.shape[0]),
        -np.ones(table.shape[0]),
    ])

    if path is not None:
        np.savetxt(path, out, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d", delimiter=",")

    return out


def mot_to_table(rows, conf_column=6):
    """
    Le linhas no formato do MOTChallenge (gt.txt ou det.txt) e devolve
    a tabela (N, 7) do repositorio, com frames 0-indexados e xyxy.

    ``gt.txt``:  frame, id, left, top, w, h, conf, class, visibility
    ``det.txt``: frame, id(-1), left, top, w, h, conf, -1, -1, -1

    ``conf_column`` diz qual coluna vai para ``conf``: 6 (a confianca,
    padrao) ou 8 (a visibilidade do gt.txt).
    """

    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))

    if rows.shape[0] == 0:
        return empty_table()

    boxes = ltwh_to_xyxy(rows[:, 2:6])

    if rows.shape[1] > conf_column:
        conf = rows[:, conf_column]
    else:
        conf = np.ones(rows.shape[0])

    return make_table(rows[:, 0] - 1, rows[:, 1], boxes, conf)
