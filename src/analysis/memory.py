import numpy as np
import torch
from torch.utils.data import DataLoader

from src.nn.boxes import ID, FRAME, X1, Y2, CONF, box_iou_matrix, cxcywh_to_xyxy
from src.nn.loss import make_box_loss
from src.nn.metrics import reacquisition_events, keep_rate_by_gap
from src.nn.models import gradient_norm_through_time
from src.nn.tracking import Tracker, StaticMotion, RNNMotion
from src.nn.train import trajectory_dataset


TRACKER = {"matching": "hungarian", "iou_threshold": 0.3, "max_age": 20}


# -------------
# Analitico
# -------------
def gradient_curve(model, seqs, length=48, n_windows=512, seed=0):
    ds = trajectory_dataset(seqs, length)
    gen = torch.Generator().manual_seed(seed)
    batch = next(iter(DataLoader(ds, batch_size=n_windows, shuffle=True, generator=gen)))

    full = batch["valid"].all(dim=1)
    batch = {k: v[full] for k, v in batch.items()}

    curve = gradient_norm_through_time(model, batch, make_box_loss("smooth_l1"), "cpu")

    return curve / curve[0]


def effective_horizon(curve, ratio=0.05):
    below = np.flatnonzero(curve < ratio)
    return int(below[0]) if below.size else len(curve)


def _iou_rows(a, b):
    a, b = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    union = np.prod(a[:, 2:] - a[:, :2], axis=1) + np.prod(b[:, 2:] - b[:, :2], axis=1) - inter
    return inter / np.maximum(union, 1e-12)


@torch.no_grad()
def free_running_iou(model, seqs, warmup=8, horizon=40):
    ds = trajectory_dataset(seqs, warmup + horizon)
    batch = next(iter(DataLoader(ds, batch_size=len(ds))))
    full = batch["valid"].all(dim=1)
    boxes = batch["boxes"][full]

    target = boxes[:, warmup:].numpy()

    if model is None:
        pred = boxes[:, warmup - 1:warmup].numpy().repeat(horizon, axis=1)
    else:
        observed = torch.ones(*boxes.shape[:2], 1)
        observed[:, warmup:] = 0.0
        pred = model.eval()(boxes, observed)["box"][:, warmup - 1:-1].numpy()

    return np.array([_iou_rows(pred[:, k], target[:, k]).mean() for k in range(horizon)])


# ----------
# Empirico 
# ----------
def run_tracker(seq, model=None, detector="SDP", min_score=0.9, detections=None, **tracker_kwargs):
    det = detections if detections is not None else seq.public_detections(detector, min_score=min_score)
    motion = StaticMotion() if model is None else RNNMotion(model, seq.height, seq.fps)

    tracker = Tracker(motion, **{**TRACKER, **tracker_kwargs})
    pred = tracker.run(det, frames=range(seq.n_frames))

    return pred, tracker.predicted_boxes_table()


def empirical_horizon(pred, gt, bins=(1, 2, 3, 5, 10, 15, 20, 30, 50, 10**9)):
    events = reacquisition_events(pred, gt)
    rates = keep_rate_by_gap([e for e in events if e["gap"] > 0], bins=bins)

    return events, rates, crossing(rates)


def crossing(rates, level=0.5):
    centers = [(r["gap_lo"] + (r["gap_hi"] if r["gap_hi"] is not None else 2 * r["gap_lo"])) / 2 for r in rates]
    keep = [r["keep_rate"] for r in rates]

    for k in range(1, len(keep)):
        if keep[k] < level <= keep[k - 1]:
            frac = (keep[k - 1] - level) / (keep[k - 1] - keep[k])
            return float(centers[k - 1] + frac * (centers[k] - centers[k - 1]))

    return float(centers[0]) if keep and keep[0] < level else float("nan")


def keep_rate_between(events, lo, hi):
    sel = [e["kept"] for e in events if lo <= e["gap"] < hi]
    return float(np.mean(sel)) if sel else float("nan"), len(sel)


# -------------------
# Galeria de falhas
# -------------------
def _gap_visibility(gt, gid, t0, t1):
    rows = gt[(gt[:, ID] == gid) & (gt[:, FRAME] >= t0) & (gt[:, FRAME] < t1)]
    return rows[:, CONF]


def pick_failures(events_by_seq, seqs, max_gap=60, max_mean_vis=0.3):
    def occluded(e, s):
        vis = _gap_visibility(seqs[s].gt, e["gt_id"], e["frame"] - e["gap"], e["frame"])
        return len(vis) > 0 and vis.mean() < max_mean_vis

    lost = [{**e, "seq": s} for s, evs in events_by_seq.items() for e in evs
            if not e["kept"] and 0 < e["gap"] <= max_gap and occluded(e, s)]
    lost.sort(key=lambda e: -e["gap"])

    picks = []

    long_gaps = [e for e in lost if e["gap"] > TRACKER["max_age"]]
    if long_gaps:
        picks.append({**long_gaps[0], "kind": "oclusao longa"})

    medium = [e for e in lost if 3 <= e["gap"] < TRACKER["max_age"]]
    if medium:
        picks.append({**medium[0], "kind": "buraco medio"})

    for s, evs in events_by_seq.items():
        owner = {}
        for e in sorted(evs, key=lambda e: e["frame"]):
            owner.setdefault(e["before"], e["gt_id"])
            if not e["kept"] and e["gap"] == 0 and owner.get(e["after"], e["gt_id"]) != e["gt_id"]:
                picks.append({**e, "seq": s, "kind": "troca", "other_gt": owner[e["after"]]})
                return picks
            owner.setdefault(e["after"], e["gt_id"])

    return picks


def diagnose(case, seq, pred, predicted, max_age=TRACKER["max_age"]):
    gt = seq.gt
    gid, t1, gap = case["gt_id"], case["frame"], case["gap"]
    t0 = t1 - gap

    person = gt[gt[:, ID] == gid]
    in_gap = person[(person[:, FRAME] >= t0) & (person[:, FRAME] < t1)]
    at_t1 = person[person[:, FRAME] == t1][0, X1:Y2 + 1]

    old = predicted[(predicted[:, ID] == case["before"]) & (predicted[:, FRAME] == t1)]
    pred_iou = float(box_iou_matrix(old[:, X1:Y2 + 1], at_t1[None])[0, 0]) if len(old) else float("nan")

    return {
        "seq": case["seq"],
        "kind": case["kind"],
        "gt_id": int(gid),
        "frame": int(t1),
        "gap": int(gap),
        "min_vis": float(in_gap[:, CONF].min()) if len(in_gap) else float("nan"),
        "mean_vis": float(in_gap[:, CONF].mean()) if len(in_gap) else float("nan"),
        "id_before": int(case["before"]),
        "id_after": int(case["after"]),
        "track_alive": bool(len(old)),
        "died_by_max_age": bool(gap > max_age),
        "pred_iou": pred_iou,
        "visible_in_gap": int((in_gap[:, CONF] >= 0.5).sum()),
        "other_gt": case.get("other_gt"),
    }
