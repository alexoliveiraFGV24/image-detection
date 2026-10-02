import numpy as np
import pandas as pd

from src.dataset.mot17 import evaluate_sequence, remove_distractor_matches
from src.nn.boxes import FRAME, X1, X2, Y1, Y2, CONF, clip_boxes
from src.nn.metrics import mean_average_precision


LEVELS = {
    "limpo": (0.0, 0.0, 0.0),
    "leve": (0.1, 0.02, 0.5),
    "media": (0.2, 0.05, 1.5),
    "pesada": (0.4, 0.10, 3.0),
}


def degrade_detections(det, drop=0.0, noise=0.0, fp_per_frame=0.0, image_size=None, n_frames=None, seed=0):
    rng = np.random.default_rng(seed)
    det = np.asarray(det, dtype=np.float64).copy()

    det = det[rng.uniform(size=len(det)) >= drop]

    if noise > 0 and len(det):
        w = det[:, X2] - det[:, X1]
        h = det[:, Y2] - det[:, Y1]
        det[:, X1:Y2 + 1] += rng.normal(size=(len(det), 4)) * noise * np.stack([w, h, w, h], axis=1)
        det[:, X2] = np.maximum(det[:, X2], det[:, X1] + 1.0)
        det[:, Y2] = np.maximum(det[:, Y2], det[:, Y1] + 1.0)

    if fp_per_frame > 0 and len(det):
        n = rng.poisson(fp_per_frame, size=n_frames)
        frames = np.repeat(np.arange(n_frames), n)
        donors = det[rng.integers(0, len(det), size=len(frames))]
        w = donors[:, X2] - donors[:, X1]
        h = donors[:, Y2] - donors[:, Y1]
        x1 = rng.uniform(size=len(frames)) * np.maximum(image_size[0] - w, 0)
        y1 = rng.uniform(size=len(frames)) * np.maximum(image_size[1] - h, 0)
        fp = np.stack([frames, -np.ones(len(frames)), x1, y1, x1 + w, y1 + h, donors[:, CONF]], axis=1)
        det = np.concatenate([det, fp])

    det[:, X1:Y2 + 1] = clip_boxes(det[:, X1:Y2 + 1], *image_size)

    return det[np.argsort(det[:, FRAME], kind="stable")]


def stress_test(seqs, trackers, levels=LEVELS, seeds=(0, 1, 2), detector="SDP", min_score=0.9):
    rows = []

    for seq in seqs:
        clean = seq.public_detections(detector, min_score=min_score)
        for level, (drop, noise, fp) in levels.items():
            for seed in (seeds if level != "limpo" else seeds[:1]):
                det = degrade_detections(clean, drop, noise, fp, seq.image_size, seq.n_frames, seed=seed)
                map_det = mean_average_precision(remove_distractor_matches(det, seq.gt_all), seq.gt)[0]
                for name, run in trackers.items():
                    r = evaluate_sequence(run(seq, det), seq)
                    rows.append({
                        "sequence": seq.name, "level": level, "seed": seed, "tracker": name,
                        "mAP_det": map_det, "mAP_track": r["mAP"], "IDF1": r["IDF1"],
                        "IDSW_per_gt_id": r["IDSW_per_gt_id"], "id_ratio": r["id_ratio"],
                    })

    return pd.DataFrame(rows)


def summarize(df, levels=LEVELS):
    mean = df.groupby(["tracker", "level"])[["mAP_det", "mAP_track", "IDF1", "IDSW_per_gt_id", "id_ratio"]].mean()
    mean = mean.reindex(pd.MultiIndex.from_product([mean.index.levels[0], list(levels)]))

    per_seed = df.groupby(["tracker", "level", "seed"])["IDF1"].mean()
    mean["IDF1_std"] = per_seed.groupby(["tracker", "level"]).std().fillna(0.0)

    out = []
    for tracker, block in mean.groupby(level=0):
        base = block.xs("limpo", level=1).iloc[0]
        block = block.copy()
        block["drop_mAP_det"] = 1 - block["mAP_det"] / base["mAP_det"]
        block["drop_IDF1"] = 1 - block["IDF1"] / base["IDF1"]
        block["amplificacao"] = block["drop_IDF1"] / block["drop_mAP_det"].replace(0, np.nan)
        out.append(block)

    return pd.concat(out)
