"""
Figuras. Identidades sempre coloridas de forma CONSISTENTE: a cor de um
id e funcao so do id (``id_color``), entao a mesma pessoa tem a mesma
cor em todos os quadros, em todas as figuras e no video da inferencia.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.nn.boxes import FRAME, ID, X1, Y1, X2, Y2, CONF, split_by_frame


_GOLDEN = 0.6180339887498949


def id_color(track_id):
    """
    Cor RGB (tupla em [0, 1]) determinada so pelo id. O matiz anda pela
    razao aurea, entao ids proximos (1, 2, 3...) ficam com cores bem
    diferentes e a paleta nao se repete -- no MOT17 ha centenas de ids
    previstos, e uma paleta de 20 cores esconderia trocas de id.
    """

    import colorsys

    track_id = int(track_id)

    if track_id < 0:
        return (0.6, 0.6, 0.6)

    hue = (track_id * _GOLDEN) % 1.0
    saturation = 0.65 + 0.35 * ((track_id // 3) % 2)
    value = 0.95 - 0.25 * ((track_id // 5) % 2)

    return colorsys.hsv_to_rgb(hue, saturation, value)


def draw_boxes(ax, rows, color_by_id=True, color=None, linestyle="-", linewidth=1.5, label_ids=True, alpha=1.0):
    """
    Desenha as caixas de uma tabela (N, 7) num eixo.

    Args:
        rows: linhas (N, 7) de UM quadro.
        color_by_id: cor pelo id (senao `color`).
        label_ids: escreve o id no canto da caixa.
    """

    for row in rows:

        c = id_color(row[ID]) if color_by_id else (color or "white")

        x1, y1, x2, y2 = row[X1:Y2 + 1]

        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=c, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
        ))

        if label_ids and row[ID] >= 0:
            ax.text(x1, y1 - 1, str(int(row[ID])), color=c, fontsize=7, va="bottom", ha="left",
                    bbox=dict(facecolor="black", alpha=0.4, pad=0.5, edgecolor="none"), clip_on=True)


def show_frames(video, frames, gt=None, pred=None, ncols=None, title=None, figsize_per=2.4, show_pred_ids=True):
    """
    Tira de quadros de um video sintetico, com o ground truth (linha
    cheia) e/ou a predicao (tracejada) coloridos por identidade.

    Args:
        video: SyntheticVideo (ou array (T, H, W)).
        frames: indices dos quadros a mostrar.
        gt, pred: tabelas (N, 7).
    """

    images = video.frames if hasattr(video, "frames") else video

    frames = list(frames)
    ncols = ncols or len(frames)
    nrows = int(np.ceil(len(frames) / ncols))

    gt_by_frame = split_by_frame(gt) if gt is not None else {}
    pred_by_frame = split_by_frame(pred) if pred is not None else {}

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows), squeeze=False)

    for k, t in enumerate(frames):

        ax = axes[k // ncols, k % ncols]
        ax.imshow(images[t], cmap="gray", vmin=0, vmax=1)

        if t in gt_by_frame:
            draw_boxes(ax, gt_by_frame[t], linestyle="-", label_ids=pred is None)

        if t in pred_by_frame:
            draw_boxes(ax, pred_by_frame[t], linestyle="--", linewidth=1.2, label_ids=show_pred_ids)

        ax.set_title(f"t = {t}", fontsize=9)
        ax.axis("off")

    for k in range(len(frames), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_occlusion_trajectory(video, track_id, occluder_id=None, n_snapshots=6):
    """
    A figura do requisito verificavel da Parte 0: a trajetoria de um
    objeto que some por N quadros e volta.

    Painel de cima: visibilidade x tempo (com o intervalo de oclusao
    total sombreado). Painel do meio: x e y do centro ao longo do tempo
    (linha fina onde ocluido). Painel de baixo: tira de quadros antes /
    durante / depois, com o objeto e o ocultador destacados.
    """

    vis = video.visibility_of(track_id)
    traj = video.trajectory_of(track_id)

    T = video.n_frames
    t = np.arange(T)

    intervals = video.occlusion_intervals(0.0).get(track_id, [])

    fig = plt.figure(figsize=(2.2 * n_snapshots, 7.5))
    grid = fig.add_gridspec(3, n_snapshots, height_ratios=[1.0, 1.0, 1.4])

    # --- visibilidade
    ax = fig.add_subplot(grid[0, :])
    ax.plot(t, vis, color=id_color(track_id), linewidth=2)
    ax.set_ylabel("visibilidade")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, T - 1)
    ax.grid(alpha=0.3)

    for (s, e) in intervals:
        ax.axvspan(s - 0.5, e + 0.5, color="red", alpha=0.15)
        ax.text((s + e) / 2, 0.5, f"some por {e - s + 1} quadros", ha="center", va="center", color="darkred")

    ax.set_title(f"objeto {track_id}" + (f" (ocultado por {occluder_id})" if occluder_id else ""))

    # --- centro x, y
    ax = fig.add_subplot(grid[1, :])
    cx = (traj[:, 0] + traj[:, 2]) / 2
    cy = (traj[:, 1] + traj[:, 3]) / 2
    visible = vis > 0

    ax.plot(t, cx, color="tab:blue", alpha=0.3)
    ax.plot(t, cy, color="tab:orange", alpha=0.3)
    ax.plot(t[visible], cx[visible], ".", color="tab:blue", label="cx (visivel)")
    ax.plot(t[visible], cy[visible], ".", color="tab:orange", label="cy (visivel)")
    ax.set_ylabel("centro (px)")
    ax.set_xlim(0, T - 1)
    ax.set_xlabel("quadro")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    for (s, e) in intervals:
        ax.axvspan(s - 0.5, e + 0.5, color="red", alpha=0.15)

    # --- snapshots
    if intervals:
        s, e = intervals[0]
        snaps = np.unique(np.clip(np.round(np.linspace(max(s - 6, 0), min(e + 6, T - 1), n_snapshots)), 0, T - 1).astype(int))
    else:
        snaps = np.linspace(0, T - 1, n_snapshots).astype(int)

    gt_by_frame = split_by_frame(video.gt)

    for k, tt in enumerate(snaps):

        ax = fig.add_subplot(grid[2, k])
        ax.imshow(video.frames[tt], cmap="gray", vmin=0, vmax=1)

        rows = gt_by_frame.get(int(tt), np.zeros((0, 7)))
        keep = np.isin(rows[:, ID], [track_id] + ([occluder_id] if occluder_id else []))
        draw_boxes(ax, rows[keep], linewidth=2)

        v = vis[tt]
        ax.set_title(f"t={tt}  vis={v:.2f}", fontsize=9, color="darkred" if v == 0 else "black")
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_detector_simulation(video, t, gt, det, title=None):
    """Um quadro com GT (linha cheia, colorido) e deteccoes simuladas (tracejado cinza)."""

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    gt_rows = split_by_frame(gt).get(t, np.zeros((0, 7)))
    det_rows = split_by_frame(det).get(t, np.zeros((0, 7)))

    axes[0].imshow(video.frames[t], cmap="gray", vmin=0, vmax=1)
    draw_boxes(axes[0], gt_rows)
    axes[0].set_title(f"ground truth  (t={t}, {len(gt_rows)} caixas)")
    axes[0].axis("off")

    axes[1].imshow(video.frames[t], cmap="gray", vmin=0, vmax=1)
    draw_boxes(axes[1], gt_rows, color_by_id=False, color="lime", linewidth=0.8, label_ids=False, alpha=0.5)
    draw_boxes(axes[1], det_rows, color_by_id=False, color="red", linestyle="--", label_ids=False)

    for row in det_rows:
        axes[1].text(row[X2], row[Y2], f"{row[CONF]:.2f}", color="red", fontsize=6, ha="right", va="top")

    axes[1].set_title(f"detector simulado  ({len(det_rows)} deteccoes)")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_breakdown(results, knob, metrics=("IDF1", "IDSW_per_gt_id", "id_ratio"), hue=None,
                   title=None, ax_labels=None, logx=False, ncols=None):
    """
    Curvas metrica x valor do botao do gerador, com media +- desvio
    sobre as seeds. E o "ensaio da Parte 1": onde o baseline quebra.

    Com metrics=("IDF1", "mAP", "id_ratio", "IDSW_per_gt_id") e ncols=2
    sai o grafico de dois paineis do enunciado: em cima mAP por quadro
    e IDF1, embaixo #ids previstos / #verdadeiros e switches por id.

    Args:
        results: DataFrame com colunas [knob, seed, *metrics] (+ hue).
        knob: nome da coluna do botao (ex.: "n_objects").
        hue: coluna categorica -> uma curva por valor (ex.: "tracker").
        ncols: paineis por linha (padrao: todos numa linha).
    """

    n = len(metrics)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax in axes[n:]:
        ax.axis("off")

    series = [(None, results)] if hue is None else list(results.groupby(hue))

    for ax, metric in zip(axes, metrics):

        for name, part in series:

            grouped = part.groupby(knob)[metric]
            mean = grouped.mean()
            std = grouped.std().fillna(0.0)

            ax.errorbar(mean.index, mean.values, yerr=std.values, marker="o", capsize=3,
                        label=None if name is None else str(name))

        ax.set_xlabel(ax_labels.get(knob, knob) if ax_labels else knob)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)

        if logx:
            ax.set_xscale("log")

        if metric in ("IDF1", "mAP"):
            ax.set_ylim(0, 1.05)

        if hue is not None and ax is axes[0]:
            ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_tracks_timeline(pred, gt, iou_threshold=0.5, ax=None, title=None, gt_ids=None, frames=None):
    """
    Linha do tempo das identidades: uma faixa por id verdadeiro, com a
    cor do id PREVISTO que estava casado em cada quadro. Trocas de cor
    dentro de uma faixa sao ID switches; cinza claro = presente no GT
    sem par; branco = ausente do GT. Rasterizada (imshow), entao aguenta
    as ~80 identidades x 1000 quadros do MOT17.

    Args:
        gt_ids: subconjunto de ids verdadeiros (padrao: todos).
        frames: (inicio, fim) inclusivo (padrao: todos os quadros do GT).
    """

    from src.nn.metrics import frame_matching

    matches, _ = frame_matching(pred, gt, iou_threshold)

    gt = np.asarray(gt)
    all_ids = sorted(np.unique(gt[:, ID]).astype(int))
    gt_ids = all_ids if gt_ids is None else list(gt_ids)
    row_of = {g: r for r, g in enumerate(gt_ids)}

    gt_frames = gt[:, 0].astype(int)
    f0, f1 = (gt_frames.min(), gt_frames.max()) if frames is None else frames

    image = np.ones((len(gt_ids), f1 - f0 + 1, 3))

    for row in gt[(gt_frames >= f0) & (gt_frames <= f1)]:

        gid = int(row[ID])

        if gid not in row_of:
            continue

        t = int(row[0])
        pid = matches.get(t, {}).get(gid)

        image[row_of[gid], t - f0] = id_color(pid) if pid is not None else (0.85, 0.85, 0.85)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, min(0.18 * len(gt_ids) + 1.2, 14)))
    else:
        fig = ax.figure

    ax.imshow(image, aspect="auto", interpolation="nearest",
              extent=(f0 - 0.5, f1 + 0.5, len(gt_ids) - 0.5, -0.5))

    if len(gt_ids) <= 40:
        ax.set_yticks(range(len(gt_ids)))
        ax.set_yticklabels([f"gt {g}" for g in gt_ids], fontsize=7)
    else:
        ax.set_ylabel("identidade verdadeira")

    ax.set_xlabel("quadro")

    if title:
        ax.set_title(title)

    return fig


def plot_decoupling(results, order, sources=None, axis_label=None, axis_values=None, title=None,
                    highlight=None):
    """
    O grafico obrigatorio da Parte 1: o DESCOLAMENTO entre a qualidade
    por quadro e a identidade no tempo, em dois paineis sobre as MESMAS
    sequencias, ordenadas pelo eixo de dificuldade.

        em cima   mAP por quadro (das deteccoes que entram no rastreador)
                  e IDF1
        embaixo   #ids previstos / #ids verdadeiros e ID switches por
                  identidade verdadeira

    Args:
        results: DataFrame com colunas sequence, source, mAP, IDF1,
            id_ratio, IDSW_per_gt_id.
        order: lista de sequencias na ordem do eixo (facil -> dificil).
        sources: fontes de deteccao a desenhar (padrao: todas).
        axis_label / axis_values: nome e valor do eixo para os rotulos.
        highlight: sequencias a destacar (ex.: as de validacao).
    """

    sources = list(results["source"].unique()) if sources is None else sources
    styles = ["-", "--", ":", "-."]

    x = np.arange(len(order))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    bottom_right = bottom.twinx()

    for k, source in enumerate(sources):

        part = results[results["source"] == source].set_index("sequence").loc[order]
        ls = styles[k % len(styles)]
        suffix = f" ({source})" if len(sources) > 1 else ""

        top.plot(x, part["mAP"], ls, marker="s", color="tab:gray", label="mAP por quadro" + suffix)
        top.plot(x, part["IDF1"], ls, marker="o", color="tab:blue", label="IDF1" + suffix)

        bottom.plot(x, part["id_ratio"], ls, marker="^", color="tab:purple",
                    label="#ids previstos / #verdadeiros" + suffix)
        bottom_right.plot(x, part["IDSW_per_gt_id"], ls, marker="v", color="tab:red",
                          label="ID switches / id verdadeiro" + suffix)

    top.set_ylim(0, 1)
    top.set_ylabel("mAP  /  IDF1")
    top.grid(alpha=0.3)
    top.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))

    bottom.axhline(1.0, color="tab:purple", alpha=0.3, linewidth=1)
    bottom.set_ylabel("#ids previstos / #verdadeiros", color="tab:purple")
    bottom_right.set_ylabel("ID switches / id verdadeiro", color="tab:red")
    bottom.set_ylim(bottom=0)
    bottom_right.set_ylim(bottom=0)
    bottom.grid(alpha=0.3)

    handles = bottom.get_legend_handles_labels()[0] + bottom_right.get_legend_handles_labels()[0]
    labels = bottom.get_legend_handles_labels()[1] + bottom_right.get_legend_handles_labels()[1]
    bottom.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(1.09, 0.5))

    ticks = []
    for i, name in enumerate(order):
        label = name.replace("MOT17-", "")
        if axis_values is not None:
            label += f"\n{axis_values[i]}"
        ticks.append(label)

    bottom.set_xticks(x)
    bottom.set_xticklabels(ticks)

    if highlight:
        for i, name in enumerate(order):
            if name in highlight:
                for ax in (top, bottom):
                    ax.axvspan(i - 0.4, i + 0.4, color="gold", alpha=0.15, linewidth=0)

    bottom.set_xlabel(f"sequencia (ordenada por {axis_label})" if axis_label else "sequencia")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_keep_rate(curves, max_age=None, ax=None, title=None):
    """
    Fracao das recapturas que mantem o id, por duracao do buraco
    (src/nn/metrics.py::keep_rate_by_gap).

    Args:
        curves: dict rotulo -> lista de {gap_lo, gap_hi, n, keep_rate}.
        max_age: desenha a linha vertical do `max_age` do rastreador.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    for label, rows in curves.items():

        centers = [r["gap_lo"] if r["gap_hi"] is None else (r["gap_lo"] + r["gap_hi"]) / 2 for r in rows]
        centers = [max(c, 0.5) for c in centers]
        ax.plot(centers, [r["keep_rate"] for r in rows], marker="o", label=label)

    if max_age is not None:
        ax.axvline(max_age + 0.5, color="k", linestyle="--", alpha=0.5)
        ax.text(max_age + 0.7, 0.05, f"max_age = {max_age}", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("duracao do buraco (quadros sem par, escala log; 0 plotado em 0,5)")
    ax.set_ylabel("fracao que mantem o id")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if title:
        ax.set_title(title)

    return fig


def show_sequence_frames(seq, frames, gt=None, pred=None, crop=None, ncols=None, figsize_per=3.2,
                         title=None, show_pred_ids=True):
    """
    Quadros reais de uma sequencia do MOT17 com o GT (linha cheia) e/ou
    a predicao (tracejada), coloridos por identidade.

    Args:
        seq: MOT17Sequence com quadros disponiveis.
        frames: indices (0-indexados).
        crop: (x1, y1, x2, y2) para dar zoom numa regiao.
    """

    frames = list(frames)
    ncols = ncols or len(frames)
    nrows = int(np.ceil(len(frames) / ncols))

    gt_by_frame = split_by_frame(gt) if gt is not None else {}
    pred_by_frame = split_by_frame(pred) if pred is not None else {}

    x1, y1, x2, y2 = crop if crop is not None else (0, 0, seq.width, seq.height)
    aspect = (y2 - y1) / (x2 - x1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * aspect * nrows + 0.4),
                             squeeze=False)

    for k, t in enumerate(frames):

        ax = axes[k // ncols, k % ncols]
        ax.imshow(seq.image(t))

        if t in gt_by_frame:
            draw_boxes(ax, gt_by_frame[t], linestyle="-", label_ids=pred is None, linewidth=1.2)

        if t in pred_by_frame:
            draw_boxes(ax, pred_by_frame[t], linestyle="--", linewidth=1.5, label_ids=show_pred_ids)

        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
        ax.set_title(f"t = {t}", fontsize=9)
        ax.axis("off")

    for k in range(len(frames), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_gradient_curves(curves, ratio=0.05, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    for label, curve in curves.items():
        k = np.arange(len(curve))
        ax.semilogy(k, np.maximum(curve, 1e-8), marker=".", label=label)

    ax.axhline(ratio, color="k", linestyle="--", alpha=0.5)
    ax.text(0.5, ratio * 1.2, f"{ratio:.0%} de k = 0", fontsize=8)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("k (passos para tras)")
    ax.set_ylabel(r"$\|\partial L_t / \partial h_{t-k}\|$ (relativa a k = 0)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    if title:
        ax.set_title(title)

    return fig


def plot_memory_horizon(rates, occlusions, max_age=None, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    occlusions = np.asarray(occlusions)
    hist_ax = ax.twinx()
    edges = np.arange(0, 64, 4)
    hist_ax.hist(np.clip(occlusions, 0, edges[-1] - 1), bins=edges, color="0.8", alpha=0.6,
                 label="oclusoes do GT")
    hist_ax.set_ylabel("oclusoes do GT (contagem)", color="0.4")
    ax.set_zorder(hist_ax.get_zorder() + 1)
    ax.patch.set_visible(False)

    for label, rows in rates.items():
        centers = [r["gap_lo"] + 2 if r["gap_hi"] is None else (r["gap_lo"] + r["gap_hi"]) / 2 for r in rows]
        ax.plot(centers, [r["keep_rate"] for r in rows], marker="o", label=label)

    if max_age is not None:
        ax.axvline(max_age + 0.5, color="k", linestyle="--", alpha=0.5)
        ax.text(max_age + 1, 0.9, f"max_age = {max_age}", fontsize=8)

    ax.set_xlim(0, edges[-1])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("duracao do buraco / da oclusao (quadros; >= 60 na ultima barra)")
    ax.set_ylabel("fracao das recapturas que mantem o id")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    if title:
        ax.set_title(title)

    return fig


def plot_failure_case(seq, case, pred, predicted, n_frames=6, margin=60, title=None):
    gt = seq.gt
    gid, t1, gap = case["gt_id"], case["frame"], case["gap"]
    t0 = t1 - gap

    if gap > 0:
        inside = np.linspace(t0, t1 - 1, max(1, min(gap, n_frames - 3))).round().astype(int)
        frames = sorted(set([t0 - 1, *inside.tolist(), t1, min(t1 + 3, seq.n_frames - 1)]))
    else:
        frames = list(range(max(t1 - 3 * (n_frames // 2), 0), t1 + 3 * (n_frames - n_frames // 2), 3))

    focus_ids = [gid] + ([case["other_gt"]] if "other_gt" in case else [])
    focus = gt[np.isin(gt[:, ID], focus_ids) & np.isin(gt[:, FRAME], frames)]
    x1, y1 = focus[:, X1].min() - margin, focus[:, Y1].min() - margin
    x2, y2 = focus[:, X2].max() + margin, focus[:, Y2].max() + margin
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, seq.width), min(y2, seq.height)

    gt_by = split_by_frame(gt)
    pred_by = split_by_frame(pred)
    predicted_old = predicted[predicted[:, ID] == case["before"]]
    predicted_by = split_by_frame(predicted_old)

    fig = plt.figure(figsize=(2.6 * len(frames), 6.2))
    grid = fig.add_gridspec(2, len(frames), height_ratios=[1.6, 1])

    for k, t in enumerate(frames):

        ax = fig.add_subplot(grid[0, k])

        if seq.has_images:
            ax.imshow(seq.image(t))
        else:
            ax.imshow(np.full((seq.height, seq.width, 3), 0.25))

        if t in gt_by:
            rows = gt_by[t]
            draw_boxes(ax, rows[~np.isin(rows[:, ID], focus_ids)], color_by_id=False, color="white",
                       linewidth=0.6, label_ids=False, alpha=0.6)
            draw_boxes(ax, rows[np.isin(rows[:, ID], focus_ids)], color_by_id=False, color="white",
                       linewidth=3, label_ids=False)

        if t in pred_by:
            draw_boxes(ax, pred_by[t], linestyle="--", linewidth=1.5)

        if t in predicted_by:
            draw_boxes(ax, predicted_by[t], linestyle=":", linewidth=2, label_ids=False)

        vis = gt[(gt[:, ID] == gid) & (gt[:, FRAME] == t), CONF]
        ax.set_title(f"t = {t}" + (f"  vis {vis[0]:.2f}" if len(vis) else ""), fontsize=8)
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    lo, hi = max(t0 - 40, 0), t1 + 20

    def cx(table, sel):
        rows = table[sel & (table[:, FRAME] >= lo) & (table[:, FRAME] <= hi)]
        rows = rows[np.argsort(rows[:, FRAME])]
        return rows[:, FRAME], (rows[:, X1] + rows[:, X2]) / 2

    ax.plot(*cx(gt, gt[:, ID] == gid), color="k", linewidth=2, label=f"GT id {gid}")
    for pid in (case["before"], case["after"]):
        ax.plot(*cx(pred, pred[:, ID] == pid), "o", markersize=3, color=id_color(pid), label=f"track {pid}")
    ax.plot(*cx(predicted, predicted[:, ID] == case["before"]), ":", color=id_color(case["before"]),
            linewidth=2, label=f"caixa prevista (track {case['before']})")

    if gap > 0:
        ax.axvspan(t0 - 0.5, t1 - 0.5, color="0.85", label="buraco")
    ax.axvline(t1, color="r", alpha=0.4)
    ax.set_xlabel("quadro")
    ax.set_ylabel("centro x (px)")
    ax.legend(fontsize=7, ncol=5, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(title or f"{seq.name} -- {case['kind']}: GT {gid}, id {case['before']} -> {case['after']}"
                 f" (buraco de {gap} quadros)", fontsize=10)
    plt.tight_layout()
    return fig


def plot_stress(summary, levels, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    x = np.arange(len(levels))
    styles = ["-", "--", ":", "-."]

    for (tracker, block), style in zip(summary.groupby(level=0), styles):
        block = block.droplevel(0).reindex(levels)
        ax.errorbar(x, block["IDF1"], yerr=block["IDF1_std"], color="C0", linestyle=style, marker="o",
                    capsize=3, label=f"IDF1 -- {tracker}")
        ax.plot(x, block["mAP_track"], color="C1", linestyle=style, marker="s", label=f"mAP trajetorias -- {tracker}")

    first = summary.xs(summary.index.get_level_values(0)[0], level=0).reindex(levels)
    ax.plot(x, first["mAP_det"], color="k", marker="^", label="mAP deteccoes (entrada)")

    ax.set_xticks(x, levels)
    ax.set_xlabel("degradacao do detector")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    return fig
