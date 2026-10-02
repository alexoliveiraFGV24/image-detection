"""
Experimentos com o MotionModel da Trilha A (Parte 3, Eixo 1 -- e os
checkpoints que a Parte 4 reaproveita).

Uma rodada = uma celula ("rnn" | "lstm" | "gru"), uma janela de BPTT
truncado T e uma seed. Ela:

    1. treina o MotionModel com o pipeline da Parte 2 (GT como alvo,
       deteccoes reais do SDP como entrada, blocos de ausencia, scheduled
       sampling, clipping), com janelas de 64 quadros e o gradiente
       truncado a cada T passos (o estado atravessa os blocos destacado);
    2. mede a previsao do quadro seguinte nas janelas de validacao, por
       passos desde a ultima observacao;
    3. mede ||dL_t / dh_{t-k}|| em funcao de k (a curva do gradiente que
       some), num lote fixo de validacao;
    4. rastreia as 7 sequencias com a regra da GRU da Parte 2 e calcula
       IDF1 / ID switches por sequencia.

Tudo vai para `out_dir/<cell>_T<T>_s<seed>.json` (+ os pesos em .pt).
Se o .json existe, a rodada e pulada -- o notebook so agrega.

Sonda de memoria (`--coast-input last_observation`, sufixo `_mem`): o
mesmo treino, mas sob oclusao entra a ultima caixa observada congelada
em vez da propria previsao. A velocidade deixa de voltar pela entrada a
cada passo e tem de atravessar o buraco no ESTADO -- a dependencia longa
que a historia do gradiente que some diz que a RNN simples nao aprende.
Sem scheduled sampling: o modelo nunca consome a propria previsao.

Mesmo orcamento de parametros: o hidden size de cada celula e o maior que
cabe no orcamento da GRU de 64 unidades da Parte 2 (13.824 parametros na
celula): RNN 113, LSTM 54, GRU 64.

    python -m src.nn.experiments --cells rnn --tbptt 4 8 16 32 --seeds 0 1 2 --threads 3
    python -m src.nn.experiments --cells rnn --tbptt 32 --coast-input last_observation
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.mot17 import load_sequences, evaluate_sequence, SEQUENCES, SPLIT
from src.dataset.trajectories import (
    extract_trajectories, attach_detections, DetectionNoise, DetectorReplay, TrajectoryDataset,
)
from src.nn.loss import make_box_loss
from src.nn.models import (
    MotionModel, train_motion_epoch, evaluate_motion, gradient_norm_through_time,
    hidden_size_for_budget, cell_parameter_count, count_parameters, save_motion_model,
)
from src.nn.optimizers import create_optimizer, create_scheduler
from src.nn.tracking import track_sequence, RNNMotion


INPUT_SIZE = 7                                              # MotionModel com dt
PARAM_BUDGET = cell_parameter_count("gru", INPUT_SIZE, 64)  # a GRU da Parte 2

# desvio robusto do erro do SDP medido na Parte 2 (so para o primeiro passo
# de uma janela em que o detector nao achou a pessoa)
SDP_SIGMA = (0.059, 0.022, 0.078, 0.041)

K_BINS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63)]

DEFAULTS = dict(window=64, stride=32, epochs=12, lr=3e-3, batch_size=64, sampling_max=0.5,
                clip_grad=1.0, beta=0.1, occlusion_prob=0.3, occlusion_len=(1, 40))

TRACK_CFG = dict(matching="hungarian", iou_threshold=0.3, max_age=40)


def k_label(lo, hi):
    return f"k={lo}" if lo == hi else f"k={lo}-{hi}"


def iou_by_k(k, iou):
    return {k_label(lo, hi): float(iou[(k >= lo) & (k <= hi)].mean()) if ((k >= lo) & (k <= hi)).any() else None
            for lo, hi in K_BINS}


# ============================================================
# DADOS
# ============================================================


class MotionData:
    """
    As trajetorias (com as deteccoes reais casadas) e os datasets de
    treino/validacao. A validacao e a MESMA para todas as rodadas (blocos
    de ausencia sorteados uma vez, com seed fixa).
    """

    def __init__(self, root="../data", detector="SDP", min_score=0.9, window=64, stride=32,
                 occlusion_prob=0.3, occlusion_len=(1, 40)):

        self.seqs = load_sequences(SEQUENCES, root=root)
        self.detections = {n: s.public_detections(detector, min_score=min_score) for n, s in self.seqs.items()}

        self.trajectories = {
            n: attach_detections(extract_trajectories(s.gt, s.height, s.fps, sequence=n), s.gt,
                                 self.detections[n], s.height)
            for n, s in self.seqs.items()
        }

        self.replay = DetectorReplay(occlusion_prob=occlusion_prob, occlusion_len=occlusion_len,
                                     fallback=DetectionNoise(sigma=SDP_SIGMA))

        self.window = window
        self.stride = stride

        self.val_ds = TrajectoryDataset(self._traj(SPLIT["val"]), length=64, stride=32,
                                        noise=self.replay, fixed=True, seed=1000)

    def _traj(self, names):
        return sum((self.trajectories[n] for n in names), [])

    def train_dataset(self, seed):
        return TrajectoryDataset(self._traj(SPLIT["train"]), length=self.window, stride=self.stride,
                                 noise=self.replay, seed=seed)

    def val_loader(self):
        return DataLoader(self.val_ds, batch_size=128, shuffle=False)

    def gradient_batch(self, n=128):
        """Lote fixo de validacao para a curva do gradiente (as n primeiras janelas)."""
        return next(iter(DataLoader(self.val_ds, batch_size=n, shuffle=False)))


# ============================================================
# UMA RODADA
# ============================================================


def run_name(cell, tbptt, seed, coast_input="prediction"):
    return f"{cell}_T{tbptt}_s{seed}" + ("_mem" if coast_input == "last_observation" else "")


def run_config(data, cell, tbptt, seed, out_dir, device="cpu", verbose=True, coast_input="prediction",
               **overrides):
    """
    Treina e avalia uma configuracao (ver o cabecalho). Devolve o dict
    gravado no .json; se ele ja existe, so carrega.
    """

    cfg = {**DEFAULTS, **overrides}

    if coast_input == "last_observation":
        # na sonda o modelo nunca consome a propria previsao: o scheduled
        # sampling so injetaria a caixa congelada marcada como observada
        cfg["sampling_max"] = 0.0

    name = run_name(cell, tbptt, seed, coast_input)
    json_path = os.path.join(out_dir, f"{name}.json")

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as fh:
            return json.load(fh)

    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    hidden = hidden_size_for_budget(cell, INPUT_SIZE, PARAM_BUDGET)
    model = MotionModel(cell, hidden_size=hidden, coast_input=coast_input).to(device)

    loss_fn = make_box_loss("smooth_l1", beta=cfg["beta"])
    optimizer = create_optimizer("Adam", model, lr=cfg["lr"])
    scheduler = create_scheduler("cosine", optimizer, T_max=cfg["epochs"])

    train_loader = DataLoader(data.train_dataset(seed), batch_size=cfg["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = data.val_loader()

    history = []
    t_start = time.time()

    for epoch in range(cfg["epochs"]):

        sampling = cfg["sampling_max"] * epoch / max(cfg["epochs"] - 1, 1)
        t0 = time.time()

        train_loss, grad_norm = train_motion_epoch(model, train_loader, optimizer, loss_fn, device,
                                                   sampling_prob=sampling, clip_grad=cfg["clip_grad"],
                                                   tbptt=tbptt)
        scheduler.step()

        ev = evaluate_motion(model, val_loader, loss_fn, device)

        history.append({
            "epoch": epoch + 1, "sampling_prob": sampling, "train_loss": train_loss, "grad_norm": grad_norm,
            "val_loss": ev["loss"], "val_iou_observed": float(ev["iou"][ev["k"] == 0].mean()),
            "val_iou_coasting": float(ev["iou"][ev["k"] >= 1].mean()), "seconds": time.time() - t0,
        })

        if verbose:
            h = history[-1]
            print(f"[{name}] época {h['epoch']:2d} | treino {h['train_loss']:.4f} | val {h['val_loss']:.4f} | "
                  f"IoU val obs/coast {h['val_iou_observed']:.3f}/{h['val_iou_coasting']:.3f} | "
                  f"‖g‖ {h['grad_norm']:.2f} | {h['seconds']:.0f} s", flush=True)

    train_seconds = time.time() - t_start

    # --- previsao do quadro seguinte, por passos desde a ultima observacao
    ev = evaluate_motion(model, val_loader, loss_fn, device)

    # --- curva do gradiente atraves do tempo (lote fixo)
    grad_curve = gradient_norm_through_time(model, data.gradient_batch(), loss_fn, device)

    # --- rastreamento nas 7 sequencias
    t0 = time.time()
    tracking = {}

    for seq_name, s in data.seqs.items():
        pred = track_sequence(data.detections[seq_name], s.n_frames,
                              motion=RNNMotion(model, s.height, s.fps, device=device), **TRACK_CFG)
        r = evaluate_sequence(pred, s, detection_map=False)
        tracking[seq_name] = {k: r[k] for k in ["IDF1", "IDP", "IDR", "IDTP", "IDFP", "IDFN", "IDSW", "FRAG",
                                                "IDSW_per_gt_id", "n_gt_ids", "n_pred_ids", "id_ratio", "MOTA"]}

    result = {
        "name": name, "cell": cell, "tbptt": tbptt, "seed": seed, "coast_input": coast_input,
        "hidden_size": hidden,
        "cell_parameters": cell_parameter_count(cell, INPUT_SIZE, hidden),
        "parameters": count_parameters(model), "config": cfg, "track_cfg": TRACK_CFG,
        "history": history, "train_seconds": train_seconds, "track_seconds": time.time() - t0,
        "val_loss": ev["loss"], "iou_by_k": iou_by_k(ev["k"], ev["iou"]),
        "grad_norm_by_k": grad_curve.tolist(),
        "tracking": tracking,
    }

    save_motion_model(model, os.path.join(out_dir, f"{name}.pt"), history=history,
                      train_cfg={**cfg, "tbptt": tbptt, "seed": seed, "loss": "smooth_l1",
                                 "coast_input": coast_input})

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)

    return result


def load_runs(out_dir):
    """Todas as rodadas gravadas em out_dir (lista de dicts)."""

    runs = []

    if not os.path.isdir(out_dir):
        return runs

    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(out_dir, fname), encoding="utf-8") as fh:
                runs.append(json.load(fh))

    return runs


def main(argv=None):

    parser = argparse.ArgumentParser(description="Parte 3, Eixo 1: celula x BPTT truncado x seed")
    parser.add_argument("--cells", nargs="+", default=["rnn", "lstm", "gru"])
    parser.add_argument("--tbptt", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--out-dir", default="reports/results/3_ablation")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--coast-input", default="prediction", choices=["prediction", "last_observation"])

    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

    data = MotionData(root=args.root)

    for seed in args.seeds:
        for cell in args.cells:
            for tbptt in args.tbptt:
                t0 = time.time()
                r = run_config(data, cell, tbptt, seed, args.out_dir, coast_input=args.coast_input,
                               epochs=args.epochs)
                idf1 = np.mean([r["tracking"][n]["IDF1"] for n in SPLIT["train"]])
                print(f"== {r['name']}: IDF1 treino {idf1:.3f} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
