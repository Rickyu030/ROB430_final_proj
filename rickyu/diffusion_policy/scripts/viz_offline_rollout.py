#!/usr/bin/env python3
"""
Layer-3 visualization: offline rollout.

For each query point q in a demo:
    obs_history = demo_obs[q - n_obs_steps + 1 : q + 1]     # (To, ...)
    action_pred = policy.predict_action(obs_history)        # (horizon=16, 7)
    executed_pred = action_pred[start:end]                  # (n_action_steps=8, 7)
    ground_truth = demo_actions[q : q + n_action_steps]     # (n_action_steps, 7)

Plot:
    * ground-truth action per dim (solid blue)
    * stitched "executed" prediction across non-overlapping chunks (dashed red)
    * every 16-step horizon fan-out as thin green lines
    * per-dim MSE

No simulator required. Uses GPU if available.

Usage:
    python scripts/viz_offline_rollout.py \
        --checkpoint data/outputs/.../checkpoints/latest.ckpt \
        --hdf5 data/libero/spatial_one_dp_demo.hdf5 \
        --demo 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import dill
import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

AG_CANDIDATES = ("agentview_image", "agentview_rgb")

DIM_LABELS = [
    "dx (norm)", "dy (norm)", "dz (norm)",
    "drx (norm)", "dry (norm)", "drz (norm)",
    "gripper",
]


def load_policy(checkpoint: Path, device: torch.device):
    print(f"[load] {checkpoint}")
    t0 = time.time()
    with open(checkpoint, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="/tmp/dp_rollout_tmp")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s")
    print(f"       horizon={cfg.horizon}  n_obs_steps={cfg.n_obs_steps}  "
          f"n_action_steps={cfg.n_action_steps}")
    return policy, cfg


def build_obs_window(
    demo_grp: h5py.Group, rgb_keys: list[str], lowdim_keys: list[str],
    q: int, n_obs: int,
) -> dict[str, np.ndarray]:
    """Extract (To, ...) obs for frames [q-n_obs+1 ... q], clip to 0 at start."""
    idxs = [max(0, q - n_obs + 1 + k) for k in range(n_obs)]
    out = {}
    for k in rgb_keys:
        img = np.stack([demo_grp["obs"][k][i] for i in idxs], axis=0)   # (To,H,W,3) uint8
        img = np.moveaxis(img, -1, 1).astype(np.float32) / 255.0         # (To,3,H,W)
        out[k] = img
    for k in lowdim_keys:
        arr = np.stack([demo_grp["obs"][k][i] for i in idxs], axis=0).astype(np.float32)
        out[k] = arr
    return out


def to_batch_tensors(obs_np: dict[str, np.ndarray], device):
    return {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs_np.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("-d", "--demo", type=int, default=0)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stride", type=int, default=0,
                    help="query spacing (default=n_action_steps)")
    ap.add_argument("--max-queries", type=int, default=-1,
                    help="limit query points (-1 = no limit)")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = Path(args.checkpoint).resolve()
    hdf5 = Path(args.hdf5).resolve()
    if not ckpt.is_file():
        sys.exit(f"ckpt not found: {ckpt}")
    if not hdf5.is_file():
        sys.exit(f"hdf5 not found: {hdf5}")

    policy, cfg = load_policy(ckpt, device)
    horizon = cfg.horizon
    n_obs = cfg.n_obs_steps
    n_act = cfg.n_action_steps
    stride = args.stride if args.stride > 0 else n_act

    shape_meta = cfg.task.shape_meta
    rgb_keys, lowdim_keys = [], []
    for k, v in shape_meta["obs"].items():
        if v.get("type", "low_dim") == "rgb":
            rgb_keys.append(k)
        else:
            lowdim_keys.append(k)
    print(f"[obs] rgb={rgb_keys}  lowdim={lowdim_keys}")

    with h5py.File(hdf5, "r") as f:
        demo_grp = f["data"][f"demo_{args.demo}"]
        gt_actions = np.asarray(demo_grp["actions"])  # (T, 7)
        T = gt_actions.shape[0]

        query_points = list(range(n_obs - 1, T - n_act + 1, stride))
        if args.max_queries > 0:
            query_points = query_points[:args.max_queries]
        print(f"[data] T={T}  query_points={len(query_points)}  stride={stride}")

        # Storage
        horizon_preds = []        # list of (horizon, 7) per query
        horizon_align_start = []  # trajectory index of action_pred[:, 0]
        executed_blocks = []      # list of (n_act, 7)
        executed_start = []       # trajectory index of each executed block

        t_start = time.time()
        with torch.no_grad():
            for i, q in enumerate(query_points):
                obs_np = build_obs_window(demo_grp, rgb_keys, lowdim_keys, q, n_obs)
                obs = to_batch_tensors(obs_np, device)
                result = policy.predict_action(obs)
                action_pred = result["action_pred"][0].cpu().numpy()  # (horizon, 7)
                executed = result["action"][0].cpu().numpy()          # (n_act, 7)

                horizon_preds.append(action_pred)
                # action_pred[:, k] corresponds to trajectory action at q - (n_obs-1) + k
                horizon_align_start.append(q - (n_obs - 1))
                executed_blocks.append(executed)
                # executed[0] corresponds to action at trajectory index q
                executed_start.append(q)

                if (i + 1) % 5 == 0 or i == len(query_points) - 1:
                    elapsed = time.time() - t_start
                    print(f"  [{i+1}/{len(query_points)}] q={q}  "
                          f"({elapsed:.1f}s, {elapsed/(i+1):.2f}s/query)")

    # Stitch executed blocks into a "continuous" predicted trajectory
    stitched = np.full_like(gt_actions, np.nan, dtype=np.float32)
    for start, block in zip(executed_start, executed_blocks):
        end = min(start + n_act, T)
        stitched[start:end] = block[: end - start]

    # MSE where both executed-pred and gt exist
    valid = ~np.isnan(stitched[:, 0])
    mse_per_dim = np.mean((stitched[valid] - gt_actions[valid]) ** 2, axis=0)
    mse_total = float(np.mean((stitched[valid] - gt_actions[valid]) ** 2))
    l1_per_dim = np.mean(np.abs(stitched[valid] - gt_actions[valid]), axis=0)
    # gripper exact-match accuracy (sign)
    g_gt = np.sign(gt_actions[valid, 6])
    g_pred = np.sign(stitched[valid, 6])
    g_acc = float(np.mean(g_gt == g_pred))

    print("\n[metrics]")
    print(f"  MSE total (first 6 dims + gripper): {mse_total:.4f}")
    for i, lab in enumerate(DIM_LABELS):
        print(f"  {lab:>14s}:  MSE={mse_per_dim[i]:.4f}  L1={l1_per_dim[i]:.3f}")
    print(f"  gripper sign accuracy: {g_acc*100:.1f}%")

    # ---------- plot ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t_sec = np.arange(T) / args.fps

    fig = plt.figure(figsize=(15, 11))
    gs = GridSpec(
        nrows=4, ncols=2,
        hspace=0.42, wspace=0.18,
        left=0.06, right=0.985, top=0.94, bottom=0.06,
    )
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(7)]

    for d in range(7):
        ax = axes[d]
        # ground-truth
        ax.plot(t_sec, gt_actions[:, d], color="tab:blue", lw=1.4, label="ground-truth")
        # horizon fan-out
        for start_idx, hpred in zip(horizon_align_start, horizon_preds):
            h_t = (np.arange(horizon) + start_idx) / args.fps
            ax.plot(h_t, hpred[:, d], color="tab:green",
                    lw=0.6, alpha=0.25)
        # stitched executed prediction (dashed red)
        ax.plot(t_sec, stitched[:, d], color="tab:red", lw=1.3, ls="--",
                label="policy (executed)")
        # gripper special y
        if d == 6:
            ax.set_ylim(-1.2, 1.2)
            ax.set_yticks([-1, 1])
        else:
            ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_title(
            f"{DIM_LABELS[d]}   MSE={mse_per_dim[d]:.4f}",
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        if d // 2 == 3 or d == 6:
            ax.set_xlabel("time (s)")
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)

    # 8th slot (bottom-right): bar chart of per-dim MSE + summary text
    ax_bar = fig.add_subplot(gs[3, 1])
    ax_bar.bar(range(7), mse_per_dim, color="tab:red", alpha=0.75)
    ax_bar.set_xticks(range(7))
    ax_bar.set_xticklabels(
        ["dx", "dy", "dz", "drx", "dry", "drz", "grip"], fontsize=9,
    )
    ax_bar.set_title(
        f"Per-dim MSE   |   total={mse_total:.4f}   |   grip acc={g_acc*100:.1f}%",
        fontsize=10,
    )
    ax_bar.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Offline rollout  -  demo_{args.demo}  "
        f"({ckpt.parent.parent.name})",
        fontsize=12,
    )

    out_path = (
        Path(args.out) if args.out
        else hdf5.with_suffix("").with_name(hdf5.stem + "_viz")
            / f"demo_{args.demo}_rollout.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
