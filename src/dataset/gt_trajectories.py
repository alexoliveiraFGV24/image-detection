import numpy as np
import torch
from torch.utils.data import Dataset

from src.nn.boxes import FRAME, X1, Y2, CONF, split_by_id, xyxy_to_cxcywh


def normalize_boxes(boxes_xyxy, image_size):
    w, h = (image_size, image_size) if np.isscalar(image_size) else image_size
    scale = np.array([w, h, w, h], dtype=np.float64)

    return xyxy_to_cxcywh(np.asarray(boxes_xyxy, dtype=np.float64)) / scale


def extract_trajectories(table, image_size, min_length=2):
    trajectories = []

    for track_id, rows in split_by_id(table).items():

        if len(rows) < min_length:
            continue

        trajectories.append({
            "id": track_id,
            "frames": rows[:, FRAME].astype(int),
            "boxes": normalize_boxes(rows[:, X1:Y2 + 1], image_size),
            "conf": rows[:, CONF],
        })

    return trajectories


class TrajectoryDataset(Dataset):
    def __init__(self, tables, image_size, length=16, stride=8, min_visibility=0.0, occlusion_prob=0.0, occlusion_len=(2, 8), seed=0):

        if isinstance(tables, np.ndarray):
            tables = [tables]

        image_sizes = [image_size] * len(tables)

        self.length = length
        self.min_visibility = min_visibility
        self.occlusion_prob = occlusion_prob
        self.occlusion_len = occlusion_len
        self.rng = np.random.default_rng(seed)

        self.windows = []

        for table, size in zip(tables, image_sizes):
            for traj in extract_trajectories(table, size, min_length=2):
                L = len(traj["frames"])

                starts = list(range(0, max(L - length, 0) + 1, stride))

                if starts[-1] + length < L:
                    starts.append(L - length)

                for s in starts:
                    self.windows.append((traj, s))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):

        traj, start = self.windows[index]

        frames = traj["frames"][start:start + self.length]
        boxes = traj["boxes"][start:start + self.length]
        conf = traj["conf"][start:start + self.length]

        L = len(frames)
        T = self.length

        observed = (conf >= self.min_visibility).astype(np.float64)

        # oclusao simulada
        if self.occlusion_prob > 0 and L > 3 and self.rng.uniform() < self.occlusion_prob:
            lo, hi = self.occlusion_len
            n = int(self.rng.integers(lo, hi + 1))
            s = int(self.rng.integers(1, max(2, L - n)))
            observed[s:s + n] = 0.0

        dt = np.ones(L)
        dt[1:] = np.diff(frames)

        def pad(a, shape_tail):
            out = np.zeros((T, *shape_tail), dtype=np.float32)
            out[:L] = np.asarray(a, dtype=np.float32).reshape(L, *shape_tail)
            return torch.from_numpy(out)

        valid = np.zeros(T, dtype=np.float32)
        valid[:L] = 1.0

        return {
            "boxes": pad(boxes, (4,)),
            "observed": pad(observed, (1,)),
            "conf": pad(conf, (1,)),
            "dt": pad(dt, (1,)),
            "valid": torch.from_numpy(valid),
            "id": int(traj["id"]),
        }


def collate_trajectories(items):
    out = {k: torch.stack([it[k] for it in items]) for k in ("boxes", "observed", "conf", "dt", "valid")}
    out["id"] = torch.tensor([it["id"] for it in items])
    return out
