"""
Metricas de deteccao (por quadro) e de rastreamento (por trajetoria),
todas implementadas aqui -- o enunciado proibe motmetrics / TrackEval.

Todas recebem TABELAS (N, 7) = (frame, id, x1, y1, x2, y2, conf), ver
``src/nn/boxes.py``.

Deteccao (o "instance-aware por quadro"):
    average_precision, mean_average_precision      AP / mAP@[.50:.95]

Rastreamento (o "identity-aware no tempo"):
    idf1                        atribuicao global 1-para-1 entre ids
    id_switches                 ID switches + fragmentacoes (CLEAR MOT)
    identity_count_error        erro de contagem de identidades unicas
    mota                        opcional, so para referencia
    evaluate_tracking           tudo de uma vez, num dict

Regras de matching (a mesma escolha da Parte 1 do PA1: gulosa por IoU
decrescente; a Hungarian tambem esta disponivel e e usada no IDF1, onde
a atribuicao precisa ser globalmente otima).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.nn.boxes import (
    ID, X1, Y2, CONF,
    box_iou_matrix, split_by_frame,
)


# ============================================================
# MATCHING  (uma matriz de IoU -> pares)
# ============================================================


def greedy_match(iou, iou_threshold):
    """
    Matching guloso por IoU decrescente sobre uma matriz (A, B).

    Cada linha casa com no maximo uma coluna e vice-versa; so pares com
    IoU >= limiar sao considerados. E a regra de `match_instances` do PA1.

    Returns:
        lista de pares (i, j).
    """

    iou = np.asarray(iou, dtype=np.float64)

    if iou.size == 0:
        return []

    candidates = np.argwhere(iou >= iou_threshold)

    if len(candidates) == 0:
        return []

    order = np.argsort(-iou[candidates[:, 0], candidates[:, 1]], kind="stable")

    matched_a, matched_b = set(), set()
    pairs = []

    for i, j in candidates[order]:

        if i in matched_a or j in matched_b:
            continue

        matched_a.add(i)
        matched_b.add(j)
        pairs.append((int(i), int(j)))

    return pairs


def hungarian_match(iou, iou_threshold):
    """
    Matching otimo (Hungarian, scipy) maximizando a soma de IoU sobre uma
    matriz (A, B); pares abaixo do limiar sao descartados depois.

    Returns:
        lista de pares (i, j).
    """

    iou = np.asarray(iou, dtype=np.float64)

    if iou.size == 0:
        return []

    # Pares invalidos recebem custo alto para nunca serem escolhidos
    # no lugar de um par valido; sao filtrados no final.
    cost = np.where(iou >= iou_threshold, 1.0 - iou, 1e6)

    rows, cols = linear_sum_assignment(cost)

    return [
        (int(i), int(j))
        for i, j in zip(rows, cols)
        if iou[i, j] >= iou_threshold
    ]


MATCHERS = {
    "greedy": greedy_match,
    "hungarian": hungarian_match,
}


def match_boxes(boxes_a, boxes_b, iou_threshold, method="greedy"):
    """Atalho: matriz de IoU + matcher escolhido. Devolve (pares, iou)."""

    iou = box_iou_matrix(boxes_a, boxes_b)
    return MATCHERS[method](iou, iou_threshold), iou


# ============================================================
# DETECCAO POR QUADRO  --  AP / mAP
# ============================================================


def _area_under_pr(scores, tp_flags, total_gt):
    """Area sob a curva precision-recall interpolada (slides 16-18)."""

    if len(scores) == 0 or total_gt == 0:
        return 0.0

    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp_flags, dtype=np.float64)[order]
    fp = 1.0 - tp

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall = cumulative_tp / total_gt

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    # Envelope: precisao interpolada (maximo a direita)
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    indices = np.where(recall[1:] != recall[:-1])[0]

    return float(np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]))


def average_precisions(pred, gt, thresholds=(0.5,)):
    """
    AP de deteccao (classe unica) para varios limiares de IoU de uma vez.

    Regra (a mesma do PA1, slides 16-18): ordena TODAS as deteccoes do
    video por confianca decrescente; cada deteccao casa com o ground
    truth ainda livre de maior IoU no seu quadro, se IoU >= limiar (TP),
    senao e FP. AP = area sob a curva precision-recall interpolada.

    A matriz de IoU de cada quadro e calculada UMA vez e reaproveitada
    em todos os limiares (o MOT17-04 tem 47 mil caixas verdadeiras).

    Args:
        pred: tabela (N, 7) de deteccoes; `conf` e o score.
        gt: tabela (M, 7) de ground truth.
        thresholds: limiares de IoU.

    Returns:
        {limiar: AP}
    """

    thresholds = [float(t) for t in thresholds]

    pred_by_frame = split_by_frame(pred)
    gt_by_frame = split_by_frame(gt)

    total_gt = sum(len(rows) for rows in gt_by_frame.values())

    scores = []
    flags = {t: [] for t in thresholds}

    for frame, rows in pred_by_frame.items():

        frame_scores = rows[:, CONF]
        order = np.argsort(-frame_scores, kind="stable")
        scores.extend(frame_scores[order].tolist())

        gt_rows = gt_by_frame.get(frame)

        if gt_rows is None:
            for t in thresholds:
                flags[t].extend([0] * len(order))
            continue

        iou = box_iou_matrix(rows[order, X1:Y2 + 1], gt_rows[:, X1:Y2 + 1])

        for t in thresholds:

            matched = np.zeros(iou.shape[1], dtype=bool)

            for p in range(iou.shape[0]):

                row = np.where(matched, -1.0, iou[p])
                best = int(np.argmax(row))

                if row[best] >= t:
                    matched[best] = True
                    flags[t].append(1)
                else:
                    flags[t].append(0)

    return {round(t, 2): _area_under_pr(scores, flags[t], total_gt) for t in thresholds}


def average_precision(pred, gt, iou_threshold=0.5):
    """AP de deteccao para um limiar de IoU (ver `average_precisions`)."""

    return average_precisions(pred, gt, [iou_threshold])[round(float(iou_threshold), 2)]


def mean_average_precision(pred, gt, thresholds=np.arange(0.50, 0.951, 0.05)):
    """
    mAP@[.50:.95] de deteccao (media da AP sobre os limiares de IoU).

    Returns:
        (mAP, {limiar: AP})
    """

    aps = average_precisions(pred, gt, thresholds)

    return float(np.mean(list(aps.values()))), aps


def detection_counts(pred, gt, iou_threshold=0.5, method="greedy"):
    """TP / FP / FN de deteccao somados sobre os quadros (sem identidade)."""

    pred_by_frame = split_by_frame(pred)
    gt_by_frame = split_by_frame(gt)

    tp = fp = fn = 0

    for frame in sorted(set(pred_by_frame) | set(gt_by_frame)):

        p = pred_by_frame.get(frame)
        g = gt_by_frame.get(frame)

        n_p = 0 if p is None else len(p)
        n_g = 0 if g is None else len(g)

        if n_p and n_g:
            pairs, _ = match_boxes(p[:, X1:Y2 + 1], g[:, X1:Y2 + 1], iou_threshold, method)
            m = len(pairs)
        else:
            m = 0

        tp += m
        fp += n_p - m
        fn += n_g - m

    return tp, fp, fn


# ============================================================
# RASTREAMENTO  --  IDF1
# ============================================================


def _id_index(table):
    """ids unicos de uma tabela (ordenados) e o mapa id -> posicao."""

    ids = np.unique(np.asarray(table)[:, ID]) if len(table) else np.zeros(0)
    return ids, {int(i): k for k, i in enumerate(ids)}


def id_overlap_matrix(pred, gt, iou_threshold=0.5):
    """
    Matriz (n_gt_ids, n_pred_ids) com o numero de quadros em que a
    identidade verdadeira i e a identidade prevista j coexistem com
    IoU >= limiar. E a base do IDF1.

    Returns:
        (overlap, gt_ids, pred_ids, gt_len, pred_len)
        gt_len / pred_len: numero de quadros de cada identidade.
    """

    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 7)

    gt_ids, gt_pos = _id_index(gt)
    pred_ids, pred_pos = _id_index(pred)

    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)

    gt_by_frame = split_by_frame(gt)
    pred_by_frame = split_by_frame(pred)

    for frame, g in gt_by_frame.items():

        p = pred_by_frame.get(frame)

        if p is None:
            continue

        iou = box_iou_matrix(g[:, X1:Y2 + 1], p[:, X1:Y2 + 1])

        gi = np.array([gt_pos[int(i)] for i in g[:, ID]])
        pj = np.array([pred_pos[int(j)] for j in p[:, ID]])

        hits = np.argwhere(iou >= iou_threshold)

        # np.add.at lida com repeticoes (um id repetido no mesmo quadro)
        np.add.at(overlap, (gi[hits[:, 0]], pj[hits[:, 1]]), 1.0)

    # quadros por identidade (np.unique devolve os ids ordenados, na mesma
    # ordem de _id_index)
    gt_len = np.unique(gt[:, ID], return_counts=True)[1].astype(np.float64) if len(gt) else np.zeros(0)
    pred_len = np.unique(pred[:, ID], return_counts=True)[1].astype(np.float64) if len(pred) else np.zeros(0)

    return overlap, gt_ids, pred_ids, gt_len, pred_len


def idf1(pred, gt, iou_threshold=0.5):
    """
    IDF1 (Ristani et al., 2016): atribuicao GLOBAL um-para-um entre
    identidades previstas e verdadeiras, ao longo da sequencia inteira,
    que maximiza o numero de quadros casados (IDTP).

        IDFP = deteccoes previstas - IDTP
        IDFN = deteccoes verdadeiras - IDTP
        IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)

    Uma identidade verdadeira que foi partida em duas previstas so pode
    "cobrar" uma delas; duas identidades trocadas cobram metade cada.
    E por isso que IDF1 mede identidade, e nao so deteccao.

    Args:
        pred, gt: tabelas (N, 7).
        iou_threshold: IoU minimo para um par (gt, pred) contar como
            "mesma pessoa" num quadro.

    Returns:
        dict com IDF1, IDP, IDR, IDTP, IDFP, IDFN e a atribuicao
        {gt_id: pred_id}.
    """

    overlap, gt_ids, pred_ids, gt_len, pred_len = id_overlap_matrix(pred, gt, iou_threshold)

    n_gt = float(gt_len.sum())
    n_pred = float(pred_len.sum())

    assignment = {}
    idtp = 0.0

    if overlap.size > 0:

        # Hungarian MAXIMIZANDO o numero de quadros casados. Pares com
        # overlap zero podem ser atribuidos, mas nao contribuem -- e o
        # mesmo que deixa-los sem par.
        rows, cols = linear_sum_assignment(-overlap)

        for i, j in zip(rows, cols):
            if overlap[i, j] > 0:
                assignment[int(gt_ids[i])] = int(pred_ids[j])
                idtp += overlap[i, j]

    idfp = n_pred - idtp
    idfn = n_gt - idtp

    denominator = 2 * idtp + idfp + idfn

    return {
        "IDF1": float(2 * idtp / denominator) if denominator > 0 else 1.0,
        "IDP": float(idtp / n_pred) if n_pred > 0 else 0.0,
        "IDR": float(idtp / n_gt) if n_gt > 0 else 0.0,
        "IDTP": int(idtp),
        "IDFP": int(idfp),
        "IDFN": int(idfn),
        "assignment": assignment,
    }


# ============================================================
# RASTREAMENTO  --  ID switches e fragmentacoes
# ============================================================


def frame_matching(pred, gt, iou_threshold=0.5, method="hungarian"):
    """
    Casa previsoes e ground truth QUADRO A QUADRO, com continuidade:
    um par (gt, pred) casado no quadro anterior e mantido no quadro
    atual se ainda tiver IoU >= limiar; so o resto passa pelo matcher.
    Sem isso, duas pessoas lado a lado trocariam de par por um capricho
    do IoU e gerariam switches que nao existem.

    Returns:
        dict frame -> {gt_id: pred_id} (so os casados) e
        dict frame -> (n_gt, n_pred) para a contagem de FP/FN.
    """

    pred_by_frame = split_by_frame(pred)
    gt_by_frame = split_by_frame(gt)

    frames = sorted(set(pred_by_frame) | set(gt_by_frame))

    matches = {}
    counts = {}

    previous = {}      # gt_id -> pred_id do quadro anterior

    for frame in frames:

        g = gt_by_frame.get(frame)
        p = pred_by_frame.get(frame)

        n_g = 0 if g is None else len(g)
        n_p = 0 if p is None else len(p)

        counts[frame] = (n_g, n_p)

        current = {}

        if n_g == 0 or n_p == 0:
            matches[frame] = current
            previous = current
            continue

        gt_ids = g[:, ID].astype(int)
        pred_ids = p[:, ID].astype(int)

        iou = box_iou_matrix(g[:, X1:Y2 + 1], p[:, X1:Y2 + 1])

        free_g = np.ones(n_g, dtype=bool)
        free_p = np.ones(n_p, dtype=bool)

        # 1. continuidade
        pred_index = {int(j): k for k, j in enumerate(pred_ids)}

        for a, gid in enumerate(gt_ids):

            pid = previous.get(int(gid))

            if pid is None or pid not in pred_index:
                continue

            b = pred_index[pid]

            if free_p[b] and iou[a, b] >= iou_threshold:
                current[int(gid)] = pid
                free_g[a] = False
                free_p[b] = False

        # 2. matcher no que sobrou
        ga = np.flatnonzero(free_g)
        pb = np.flatnonzero(free_p)

        if ga.size and pb.size:

            pairs = MATCHERS[method](iou[np.ix_(ga, pb)], iou_threshold)

            for i, j in pairs:
                current[int(gt_ids[ga[i]])] = int(pred_ids[pb[j]])

        matches[frame] = current
        previous = current

    return matches, counts


def id_switches(pred, gt, iou_threshold=0.5, method="hungarian"):
    """
    ID switches e fragmentacoes, contados explicitamente (CLEAR MOT):

        ID switch      a identidade verdadeira g esta casada no quadro t
                       com a previsao p, e a ULTIMA vez que g foi casada
                       (em qualquer quadro anterior) foi com p' != p.

        fragmentacao   g estava casada, ficou >= 1 quadro presente no GT
                       porem sem par, e voltou a ser casada.

    Returns:
        dict com IDSW, FRAG, TP, FP, FN, n_gt_ids, e a lista de eventos
        de switch (frame, gt_id, de, para) para a galeria de falhas.
    """

    matches, counts = frame_matching(pred, gt, iou_threshold, method)

    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 7)
    gt_by_frame = split_by_frame(gt)

    last_pred = {}          # gt_id -> ultimo pred_id casado
    was_lost = {}           # gt_id -> ficou sem par desde o ultimo casamento?

    switches = []
    fragmentations = 0

    tp = fp = fn = 0

    for frame in sorted(matches):

        current = matches[frame]
        n_g, n_p = counts[frame]

        tp += len(current)
        fp += n_p - len(current)
        fn += n_g - len(current)

        present = set(gt_by_frame[frame][:, ID].astype(int)) if frame in gt_by_frame else set()

        for gid in present:

            pid = current.get(gid)

            if pid is None:
                # presente no GT, sem par: candidata a fragmentacao
                if gid in last_pred:
                    was_lost[gid] = True
                continue

            if gid in last_pred and last_pred[gid] != pid:
                switches.append((frame, gid, last_pred[gid], pid))

            if was_lost.get(gid, False):
                fragmentations += 1
                was_lost[gid] = False

            last_pred[gid] = pid

    n_gt_ids = len(np.unique(gt[:, ID])) if len(gt) else 0

    return {
        "IDSW": len(switches),
        "FRAG": fragmentations,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "n_gt_ids": n_gt_ids,
        "IDSW_per_gt_id": len(switches) / n_gt_ids if n_gt_ids else 0.0,
        "switch_events": switches,
    }


# ============================================================
# RASTREAMENTO  --  contagem de identidades, MOTA, resumo
# ============================================================


def identity_count_error(pred, gt):
    """
    Erro de contagem de identidades unicas do video -- o analogo
    temporal do erro de contagem do PA1.

    Returns:
        dict com n_gt_ids, n_pred_ids, o erro absoluto e a razao
        previstas / verdadeiras (o eixo de baixo do grafico da Parte 1).
    """

    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 7)

    n_pred = len(np.unique(pred[:, ID])) if len(pred) else 0
    n_gt = len(np.unique(gt[:, ID])) if len(gt) else 0

    return {
        "n_gt_ids": n_gt,
        "n_pred_ids": n_pred,
        "count_error": abs(n_pred - n_gt),
        "id_ratio": n_pred / n_gt if n_gt else float("nan"),
    }


def mota(switch_stats, gt):
    """
    MOTA = 1 - (FN + FP + IDSW) / #GT. Opcional no enunciado; fica aqui
    so para referencia, a partir do resultado de `id_switches`.
    """

    n_gt = len(np.asarray(gt).reshape(-1, 7))

    if n_gt == 0:
        return float("nan")

    return 1.0 - (switch_stats["FN"] + switch_stats["FP"] + switch_stats["IDSW"]) / n_gt


def evaluate_tracking(pred, gt, iou_threshold=0.5, match_method="hungarian", detection_map=True):
    """
    Todas as metricas de uma vez, num dict plano (serializavel em JSON):

        IDF1, IDP, IDR, IDTP, IDFP, IDFN, IDSW, FRAG, IDSW_per_gt_id, n_gt_ids, n_pred_ids,
        count_error, id_ratio, MOTA, TP, FP, FN, mAP (opcional), AP@0.50
    """

    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 7)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 7)

    id_stats = idf1(pred, gt, iou_threshold)
    sw_stats = id_switches(pred, gt, iou_threshold, match_method)
    count = identity_count_error(pred, gt)

    out = {
        "IDF1": id_stats["IDF1"],
        "IDP": id_stats["IDP"],
        "IDR": id_stats["IDR"],
        "IDTP": id_stats["IDTP"],
        "IDFP": id_stats["IDFP"],
        "IDFN": id_stats["IDFN"],
        "IDSW": sw_stats["IDSW"],
        "FRAG": sw_stats["FRAG"],
        "IDSW_per_gt_id": sw_stats["IDSW_per_gt_id"],
        "n_gt_ids": count["n_gt_ids"],
        "n_pred_ids": count["n_pred_ids"],
        "count_error": count["count_error"],
        "id_ratio": count["id_ratio"],
        "MOTA": mota(sw_stats, gt),
        "TP": sw_stats["TP"],
        "FP": sw_stats["FP"],
        "FN": sw_stats["FN"],
    }

    if detection_map:
        map_value, aps = mean_average_precision(pred, gt)
        out["mAP"] = map_value
        out["AP@0.50"] = aps[0.5]

    return out


# ============================================================
# DIAGNOSTICO  --  a identidade sobrevive a um buraco?
# ============================================================


def reacquisition_events(pred, gt, iou_threshold=0.5, method="hungarian"):
    """
    Horizonte de memoria EMPIRICO de um rastreador: cada vez que uma
    identidade verdadeira volta a ser casada depois de `gap` quadros
    presente no GT porem sem par (buraco: o detector nao a achou, ou a
    associacao nao a ligou), registra se o id previsto e o MESMO de
    antes do buraco.

    Passagens sem buraco (gap = 0) tambem entram: uma troca de id de um
    quadro para o seguinte e um evento com gap 0 e kept = False.

    Returns:
        lista de dicts {gt_id, frame, gap, kept, before, after}.
    """

    matches, _ = frame_matching(pred, gt, iou_threshold, method)

    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 7)
    gt_by_frame = split_by_frame(gt)

    last_pred = {}       # gt_id -> ultimo pred_id casado
    gap = {}             # gt_id -> quadros sem par desde o ultimo casamento

    events = []

    for frame in sorted(matches):

        current = matches[frame]

        present = gt_by_frame[frame][:, ID].astype(int) if frame in gt_by_frame else []

        for gid in present:

            gid = int(gid)
            pid = current.get(gid)

            if pid is None:
                if gid in last_pred:
                    gap[gid] = gap.get(gid, 0) + 1
                continue

            if gid in last_pred:
                events.append({
                    "gt_id": gid,
                    "frame": int(frame),
                    "gap": int(gap.get(gid, 0)),
                    "kept": bool(last_pred[gid] == pid),
                    "before": int(last_pred[gid]),
                    "after": int(pid),
                })

            last_pred[gid] = pid
            gap[gid] = 0

    return events


def keep_rate_by_gap(events, bins=(0, 1, 2, 3, 5, 10, 20, 40, 80, 10**9)):
    """
    Agrega `reacquisition_events`: para cada faixa de duracao do buraco,
    a fracao de recapturas que mantiveram o id.

    Returns:
        lista de dicts {gap_lo, gap_hi, n, keep_rate}.
    """

    gaps = np.array([e["gap"] for e in events], dtype=np.int64)
    kept = np.array([e["kept"] for e in events], dtype=bool)

    out = []

    for lo, hi in zip(bins[:-1], bins[1:]):

        sel = (gaps >= lo) & (gaps < hi)

        if sel.sum() == 0:
            continue

        out.append({
            "gap_lo": int(lo),
            "gap_hi": int(hi - 1) if hi < 10**9 else None,
            "n": int(sel.sum()),
            "keep_rate": float(kept[sel].mean()),
        })

    return out
