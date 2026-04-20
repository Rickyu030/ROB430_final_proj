#!/usr/bin/env python3
"""
Plot train/val/mse curves from a diffusion_policy training run.

Usage:
    python scripts/plot_training_curves.py <run_dir_or_logs.json.txt> [-o out.png]

Defaults to the most-recent output directory under data/outputs/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_latest_run() -> Path:
    root = Path(__file__).resolve().parents[1] / "data" / "outputs"
    if not root.is_dir():
        raise SystemExit(f"No outputs under {root}")
    latest = max(root.glob("*/*/logs.json.txt"), key=lambda p: p.stat().st_mtime, default=None)
    if latest is None:
        raise SystemExit(f"No logs.json.txt under {root}")
    return latest


def load(path: Path):
    per_batch, per_epoch = [], []
    for line in path.open():
        d = json.loads(line)
        per_batch.append(d)
        if "val_loss" in d:  # end-of-epoch summary row
            per_epoch.append(d)
    return per_batch, per_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="logs.json.txt OR the run dir containing it")
    ap.add_argument("-o", "--out", default=None, help="output png (default: <run>/training_curves.png)")
    args = ap.parse_args()

    if args.path is None:
        log_path = find_latest_run()
    else:
        p = Path(args.path)
        log_path = p if p.is_file() else p / "logs.json.txt"
        if not log_path.is_file():
            sys.exit(f"Not found: {log_path}")

    per_batch, per_epoch = load(log_path)
    if not per_epoch:
        sys.exit("No epoch summaries (lines with val_loss) found.")

    out_path = Path(args.out) if args.out else log_path.parent / "training_curves.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- train loss: batch-level (light) + epoch-level (bold) ---
    ax = axes[0]
    steps = [d["global_step"] for d in per_batch]
    batch_losses = [d["train_loss"] for d in per_batch]
    ax.plot(steps, batch_losses, alpha=0.2, color="tab:blue", label="per-batch")
    ep_steps = [d["global_step"] for d in per_epoch]
    ep_train = [d["train_loss"] for d in per_epoch]
    ax.plot(ep_steps, ep_train, color="tab:blue", linewidth=2, label="epoch avg")
    ax.set_xlabel("global_step")
    ax.set_ylabel("train_loss (diffusion MSE)")
    ax.set_yscale("log")
    ax.set_title("Train loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # --- val loss ---
    ax = axes[1]
    ep_idx = [d["epoch"] for d in per_epoch]
    val = [d["val_loss"] for d in per_epoch]
    ax.plot(ep_idx, ep_train, label="train (epoch avg)", color="tab:blue")
    ax.plot(ep_idx, val, label="val", color="tab:orange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("Train vs Val")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # --- action MSE (end-to-end) ---
    ax = axes[2]
    mse_pts = [(d["epoch"], d["train_action_mse_error"]) for d in per_epoch if d.get("train_action_mse_error") is not None]
    if mse_pts:
        xs, ys = zip(*mse_pts)
        ax.plot(xs, ys, "o-", color="tab:green")
        ax.set_xlabel("epoch")
        ax.set_ylabel("action MSE (100-step DDPM vs GT)")
        ax.set_yscale("log")
        ax.set_title("Action prediction MSE")
        ax.grid(True, which="both", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no MSE samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(f"{log_path.parent.name}  ({len(per_epoch)} epochs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
