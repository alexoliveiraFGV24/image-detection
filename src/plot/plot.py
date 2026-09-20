"""
Figuras. Identidades sempre coloridas de forma CONSISTENTE: a cor de um
id e funcao so do id (``id_color``), entao a mesma pessoa tem a mesma
cor em todos os quadros, em todas as figuras e no video da inferencia.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.nn.boxes import ID, X1, X2, Y2, CONF, split_by_frame


_PALETTE = plt.get_cmap("tab20").colors


def id_color(track_id):
    """Cor RGB (tupla em [0, 1]) determinada pelo id."""

    track_id = int(track_id)

    if track_id < 0:
        return (0.6, 0.6, 0.6)

    return _PALETTE[(track_id * 7) % len(_PALETTE)]


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
                    bbox=dict(facecolor="black", alpha=0.4, pad=0.5, edgecolor="none"))


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


def plot_tracks_timeline(pred, gt, iou_threshold=0.5, ax=None, title=None):
    """
    Linha do tempo das identidades: uma faixa por id verdadeiro, com a
    cor do id PREVISTO que estava casado em cada quadro. Trocas de cor
    dentro de uma faixa sao ID switches; buracos sao quadros sem par.
    """

    from src.nn.metrics import frame_matching

    matches, _ = frame_matching(pred, gt, iou_threshold)

    gt_ids = sorted(np.unique(np.asarray(gt)[:, ID]).astype(int))
    gt_by_frame = split_by_frame(gt)

    frames = sorted(gt_by_frame)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 0.35 * len(gt_ids) + 1.2))
    else:
        fig = ax.figure

    for row, gid in enumerate(gt_ids):

        for t in frames:

            present = gid in gt_by_frame[t][:, ID].astype(int)

            if not present:
                continue

            pid = matches.get(t, {}).get(gid)

            color = id_color(pid) if pid is not None else (0.85, 0.85, 0.85)
            ax.add_patch(Rectangle((t - 0.5, row - 0.4), 1.0, 0.8, color=color, linewidth=0))

    ax.set_xlim(min(frames) - 0.5, max(frames) + 0.5)
    ax.set_ylim(-0.6, len(gt_ids) - 0.4)
    ax.set_yticks(range(len(gt_ids)))
    ax.set_yticklabels([f"gt {g}" for g in gt_ids])
    ax.set_xlabel("quadro")

    if title:
        ax.set_title(title)

    return fig
