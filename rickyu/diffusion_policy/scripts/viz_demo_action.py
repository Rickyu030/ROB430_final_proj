#!/usr/bin/env python3
"""
Layer-2 visualization: plot action trajectory of a LIBERO demo.

OSC_POSE 7-dim action over time:
    [0:3] dx, dy, dz            (normalized, *0.05 m)
    [3:6] drx, dry, drz         (normalized, *0.5 rad)
    [6]   gripper               (-1 open, +1 close)

Layout:
    Top    : 8 evenly-spaced agentview thumbnails
    Middle : 3x2 grid for dx/dy/dz and drx/dry/drz
    Bottom : gripper state (full width)
    Gripper transitions drawn as dashed vertical lines across all axes.

Usage:
    python scripts/viz_demo_action.py data/libero/spatial_one_dp_demo.hdf5 -d 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

AG_CANDIDATES = ["agentview_image", "agentview_rgb"]


def detect_agentview(obs: h5py.Group) -> str:
    for k in AG_CANDIDATES:
        if k in obs:
            return k
    raise KeyError(f"no agentview image in {list(obs.keys())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("-d", "--demo", type=int, default=0)
    ap.add_argument("-o", "--out", default=None, help="output png path")
    ap.add_argument("--fps", type=float, default=20.0, help="used for x-axis seconds")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--n-thumbs", type=int, default=8)
    args = ap.parse_args()

    hdf5 = Path(args.hdf5).resolve()
    if not hdf5.is_file():
        sys.exit(f"Not found: {hdf5}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    with h5py.File(hdf5, "r") as f:
        dk = f"demo_{args.demo}"
        if dk not in f["data"]:
            sys.exit(f"{dk} not in {hdf5}")
        grp = f["data"][dk]
        actions = np.asarray(grp["actions"])  # (T, 7)
        ag_key = detect_agentview(grp["obs"])
        imgs = np.asarray(grp["obs"][ag_key])  # (T, H, W, 3)
        T = actions.shape[0]

        import json
        pi = f["data"].attrs.get("problem_info")
        if isinstance(pi, bytes):
            pi = pi.decode("utf-8")
        try:
            lang = json.loads(pi).get("language_instruction", "") if pi else ""
        except Exception:
            lang = ""

    flip_v = not args.no_flip
    if flip_v:
        imgs = imgs[:, ::-1]

    # Gripper open->close and close->open transitions.
    g = actions[:, 6]
    g_bin = (g > 0).astype(int)  # +1 -> 1 (close), -1 -> 0 (open)
    transitions = np.where(np.diff(g_bin) != 0)[0] + 1  # frame index of change

    t_sec = np.arange(T) / args.fps

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(
        nrows=5, ncols=3,
        height_ratios=[1.1, 1, 1, 1, 0.9],
        hspace=0.45, wspace=0.22,
        left=0.06, right=0.985, top=0.92, bottom=0.07,
    )

    # ---- Thumbnail strip (row 0, spans all 3 cols) ----
    thumb_ax = fig.add_subplot(gs[0, :])
    n = min(args.n_thumbs, T)
    thumb_idx = np.linspace(0, T - 1, n, dtype=int)
    thumbs = imgs[thumb_idx]  # (n, H, W, 3)
    strip = np.concatenate(list(thumbs), axis=1)  # (H, n*W, 3)
    thumb_ax.imshow(strip)
    thumb_ax.set_xticks([])
    thumb_ax.set_yticks([])
    H, W = imgs.shape[1], imgs.shape[2]
    # Annotate frame idx under each thumbnail
    for i, idx in enumerate(thumb_idx):
        thumb_ax.text(
            i * W + W / 2, H + 4, f"t={idx/args.fps:.1f}s",
            ha="center", va="top", fontsize=8, color="black",
        )
    thumb_ax.set_title(
        f"demo_{args.demo}  |  {T} frames @ {args.fps:g} Hz ({T/args.fps:.1f}s)"
        + (f"  |  {lang}" if lang else ""),
        fontsize=11,
    )

    # ---- Shared transition line drawer ----
    def draw_transitions(ax):
        for tr in transitions:
            ax.axvline(tr / args.fps, color="red", lw=0.8, ls="--", alpha=0.55)

    labels_pos = [("dx  (*0.05 m)", 0), ("dy  (*0.05 m)", 1), ("dz  (*0.05 m)", 2)]
    labels_rot = [("drx  (*0.5 rad)", 3), ("dry  (*0.5 rad)", 4), ("drz  (*0.5 rad)", 5)]

    # ---- Row 1: position deltas ----
    for col, (lab, d) in enumerate(labels_pos):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(t_sec, actions[:, d], color="tab:blue", lw=1.2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
        draw_transitions(ax)
        if col == 0:
            ax.set_ylabel("norm action")

    # ---- Row 2: rotation deltas ----
    for col, (lab, d) in enumerate(labels_rot):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(t_sec, actions[:, d], color="tab:orange", lw=1.2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
        draw_transitions(ax)
        if col == 0:
            ax.set_ylabel("norm action")

    # ---- Row 3: |Δpos| and |Δrot| magnitudes (summary) ----
    ax_mag = fig.add_subplot(gs[3, :])
    pos_mag = np.linalg.norm(actions[:, 0:3], axis=1)
    rot_mag = np.linalg.norm(actions[:, 3:6], axis=1)
    ax_mag.plot(t_sec, pos_mag, label="|Δpos|", color="tab:blue", lw=1.4)
    ax_mag.plot(t_sec, rot_mag, label="|Δrot|", color="tab:orange", lw=1.4)
    ax_mag.set_ylabel("magnitude")
    ax_mag.grid(alpha=0.3)
    ax_mag.legend(loc="upper right", fontsize=9)
    ax_mag.set_title("End-effector command magnitude", fontsize=10)
    draw_transitions(ax_mag)

    # ---- Row 4: gripper state ----
    ax_g = fig.add_subplot(gs[4, :])
    ax_g.step(t_sec, actions[:, 6], where="post", color="tab:green", lw=1.6)
    ax_g.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_g.set_ylim(-1.2, 1.2)
    ax_g.set_yticks([-1, 1])
    ax_g.set_yticklabels(["open (-1)", "close (+1)"])
    ax_g.set_xlabel("time (s)")
    ax_g.set_title("Gripper command", fontsize=10)
    ax_g.grid(alpha=0.3)
    draw_transitions(ax_g)

    # Shared x-limit
    for ax in fig.axes[1:]:
        ax.set_xlim(0, t_sec[-1] if T > 1 else 1)

    out_path = (
        Path(args.out) if args.out
        else hdf5.with_suffix("").with_name(hdf5.stem + "_viz") / f"demo_{args.demo}_action.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")
    print(f"  T={T}  gripper transitions at frames: {transitions.tolist()}")
    print(f"  |Δpos| mean={pos_mag.mean():.3f} max={pos_mag.max():.3f}")
    print(f"  |Δrot| mean={rot_mag.mean():.3f} max={rot_mag.max():.3f}")


if __name__ == "__main__":
    main()
