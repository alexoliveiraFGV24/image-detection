"""
Trajetorias do ground truth como sequencias para a Trilha A.

``TrajectoryDataset`` recebe tabelas (N, 7) -- de videos sinteticos ou
do gt.txt do MOT17 -- e devolve janelas de comprimento fixo de cada
identidade, ja no formato que ``MotionModel`` consome:

    boxes     (T, 4)  (cx, cy, w, h) normalizadas pelo tamanho da imagem
    observed  (T, 1)  1 = observacao disponivel no passo
    conf      (T, 1)  confianca (a visibilidade, no GT)
    dt        (T, 1)  intervalo desde a observacao anterior (em quadros)
    valid     (T,)    1 nos passos reais, 0 no padding

A oclusao entra de duas formas, as duas com `observed = 0`:
  * a do proprio dado: `visibility < min_visibility` (o "ouro" do
    campo visibility do MOT17);
  * a simulada: janelas aleatorias de `occlusion_prob` / `occlusion_len`
    (augmentation), para o modelo aprender a rodar sem observacao.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.nn.boxes import FRAME, X1, Y2, CONF, split_by_id, xyxy_to_cxcywh


def normalize_boxes(boxes_xyxy, image_size):
    """(N, 4) xyxy em pixels -> (N, 4) cxcywh em [0, 1]."""

    w, h = (image_size, image_size) if np.isscalar(image_size) else image_size
    scale = np.array([w, h, w, h], dtype=np.float64)

    return xyxy_to_cxcywh(np.asarray(boxes_xyxy, dtype=np.float64)) / scale


def extract_trajectories(table, image_size, min_length=2):
    """
    Lista de trajetorias, uma por identidade, cada uma um dict com
    frames (L,), boxes (L, 4) normalizadas cxcywh, conf (L,).
    Lacunas (quadros ausentes no GT) sao mantidas como saltos em `frames`.
    """

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
    """
    Janelas de `length` passos de todas as trajetorias das tabelas.

    Args:
        tables: lista de tabelas (N, 7) (um video / sequencia cada) ou
            uma unica tabela.
        image_size: lado ou (w, h) das imagens (o mesmo para todas as
            tabelas; use um dataset por sequencia + ConcatDataset se
            variar).
        length: comprimento da janela T.
        stride: passo entre janelas de uma mesma trajetoria.
        min_visibility: abaixo disto o passo conta como NAO observado.
        occlusion_prob: probabilidade de inserir uma oclusao simulada
            numa janela; `occlusion_len`: (min, max) da sua duracao.
        seed: para a augmentation.
    """

    def __init__(self, tables, image_size, length=16, stride=8, min_visibility=0.0,
                 occlusion_prob=0.0, occlusion_len=(2, 8), seed=0):

        if isinstance(tables, np.ndarray):
            tables = [tables]

        # um tamanho para todas as tabelas; sequencias com tamanhos
        # diferentes (MOT17) viram datasets separados + ConcatDataset
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
    """Empilha os dicts (todas as janelas ja tem o mesmo T)."""

    out = {k: torch.stack([it[k] for it in items]) for k in ("boxes", "observed", "conf", "dt", "valid")}
    out["id"] = torch.tensor([it["id"] for it in items])
    return out
