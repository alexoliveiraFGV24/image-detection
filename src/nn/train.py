import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.dataset.gt_trajectories import TrajectoryDataset, collate_trajectories
from src.nn.gt_motion import GTMotionModel, train_motion_epoch, evaluate_motion
from src.nn.loss import make_box_loss


DEFAULT_CONFIG = {
    "cell": "gru",
    "hidden_size": 64,
    "length": 16,
    "occlusion_len": (2, 8),
    "occlusion_prob": 0.5,
    "epochs": 15,
    "lr": 3e-3,
    "batch_size": 128,
    "clip_grad": 1.0,
    "seed": 0,
}


CORRECTED_CONFIG = {
    "length": 32,
    "occlusion_len": (8, 24),
    "occlusion_prob": 1.0,
    "epochs": 25,
}


def trajectory_dataset(seqs, length, occlusion_len=(2, 8), occlusion_prob=0.0, seed=0):
    return ConcatDataset([
        TrajectoryDataset(s.gt, s.image_size, length=length, stride=length // 2, occlusion_prob=occlusion_prob, occlusion_len=tuple(occlusion_len), seed=seed + k)
        for k, s in enumerate(seqs)
    ])


def build_model(config):
    return GTMotionModel(cell=config["cell"], hidden_size=config["hidden_size"], use_conf=False, use_dt=False)


def train_motion(train_seqs, val_seqs=None, device="cpu", verbose=True, **overrides):
    config = {**DEFAULT_CONFIG, **overrides}

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_ds = trajectory_dataset(train_seqs, config["length"], config["occlusion_len"], config["occlusion_prob"], config["seed"])
    loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_trajectories)

    val_loader = None
    if val_seqs:
        val_ds = trajectory_dataset(val_seqs, config["length"])
        val_loader = DataLoader(val_ds, batch_size=256, collate_fn=collate_trajectories)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    loss_fn = make_box_loss("smooth_l1")

    history = []

    for epoch in range(config["epochs"]):

        loss, norm = train_motion_epoch(model, loader, optimizer, loss_fn, device, clip_grad=config["clip_grad"])
        scheduler.step()

        row = {"epoch": epoch + 1, "train_loss": loss, "grad_norm": norm}

        if val_loader is not None:
            row["val_l1"] = evaluate_motion(model, val_loader, loss_fn, device)[1]

        history.append(row)

        if verbose:
            print("  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    return model.eval(), config, history


def save_model(model, config, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["config"]


def load_or_train(path, train_seqs, val_seqs=None, device="cpu", **overrides):
    if os.path.exists(path):
        model, config = load_model(path, device)
        print(f"checkpoint carregado: {path}")
        return model, config

    model, config, _ = train_motion(train_seqs, val_seqs, device=device, **overrides)
    save_model(model, config, path)
    print(f"checkpoint salvo: {path}")
    return model, config


def main(argv=None):

    from src.dataset.mot17 import load_sequences, SPLIT

    parser = argparse.ArgumentParser(description="Treina o MotionModel (Trilha A) no GT do MOT17")
    parser.add_argument("--root", default="data")
    parser.add_argument("--out", default="reports/results/motion_gru.pt")
    parser.add_argument("--cell", default="gru")
    parser.add_argument("--length", type=int, default=DEFAULT_CONFIG["length"])
    parser.add_argument("--occlusion-max", type=int, default=DEFAULT_CONFIG["occlusion_len"][1])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    args = parser.parse_args(argv)

    seqs = load_sequences(root=args.root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config, _ = train_motion(
        [seqs[n] for n in SPLIT["train"]], [seqs[n] for n in SPLIT["val"]], device=device,
        cell=args.cell, length=args.length, epochs=args.epochs,
        occlusion_len=(2, args.occlusion_max),
    )
    save_model(model, config, args.out)
    print(f"salvo em {args.out}")


if __name__ == "__main__":
    main()
