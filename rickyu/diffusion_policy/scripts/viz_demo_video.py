#!/usr/bin/env python3
"""
Layer-1 visualization: replay human demos from a LIBERO/DP HDF5 as mp4.

Side-by-side left=agentview, right=eye-in-hand. No model involved.

Auto-detects obs keys (DP-converted vs raw LIBERO):
    DP-converted: agentview_image, robot0_eye_in_hand_image
    Raw LIBERO:   agentview_rgb,   eye_in_hand_rgb

Flips images vertically by default (LIBERO/robosuite convention).

Usage:
    python scripts/viz_demo_video.py <hdf5_path>                       # demo_0
    python scripts/viz_demo_video.py <hdf5_path> -d 3                  # demo_3
    python scripts/viz_demo_video.py <hdf5_path> --all                 # every demo
    python scripts/viz_demo_video.py <hdf5_path> -o /tmp/viz --fps 30  # custom out / fps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

# Possible (agentview_key, eye_in_hand_key) pairs in priority order.
OBS_KEY_CANDIDATES: list[Tuple[str, str]] = [
    ("agentview_image", "robot0_eye_in_hand_image"),  # DP-converted
    ("agentview_rgb", "eye_in_hand_rgb"),             # raw LIBERO
]


def detect_obs_keys(obs_group: h5py.Group) -> Tuple[str, str]:
    for a, e in OBS_KEY_CANDIDATES:
        if a in obs_group and e in obs_group:
            return a, e
    raise KeyError(
        f"Can't find agentview+eye_in_hand images. Available keys: {list(obs_group.keys())}"
    )


def stack_frames(
    ag: np.ndarray,
    eh: np.ndarray,
    flip_v: bool,
    gap_px: int = 4,
) -> np.ndarray:
    """ag, eh: (T,H,W,3) uint8 -> (T,H,W*2+gap,3) uint8 side by side."""
    assert ag.shape == eh.shape, f"camera shapes differ: {ag.shape} vs {eh.shape}"
    if flip_v:
        ag = ag[:, ::-1]
        eh = eh[:, ::-1]
    T, H, W, _ = ag.shape
    gap = np.zeros((T, H, gap_px, 3), dtype=np.uint8)
    return np.concatenate([ag, gap, eh], axis=2)


def task_title(f: h5py.File) -> str:
    """Pull language instruction out of data.attrs.problem_info if present."""
    import json

    pi = f["data"].attrs.get("problem_info")
    if pi is None:
        return ""
    if isinstance(pi, bytes):
        pi = pi.decode("utf-8")
    try:
        return json.loads(pi).get("language_instruction", "")
    except Exception:
        return ""


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    """Top-left corner white text. Uses cv2 if available, else PIL fallback."""
    if not text:
        return frame
    try:
        import cv2

        img = frame.copy()
        # Draw shadow then white text
        cv2.putText(
            img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        return img
    except Exception:
        return frame


def write_mp4(frames: np.ndarray, out_path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 disables "multiple of 16" padding warning on 128x260
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", help="Path to .hdf5 file (DP-converted or raw LIBERO)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("-d", "--demo", type=int, default=0, help="demo index (default 0)")
    grp.add_argument("--all", action="store_true", help="export every demo")
    ap.add_argument("-o", "--out", default=None, help="output dir (default: <hdf5>_viz/)")
    ap.add_argument("--fps", type=int, default=20, help="playback fps (LIBERO ~20)")
    ap.add_argument("--no-flip", action="store_true", help="don't flip images vertically")
    ap.add_argument("--no-title", action="store_true", help="don't overlay language instruction")
    args = ap.parse_args()

    hdf5_path = Path(args.hdf5).resolve()
    if not hdf5_path.is_file():
        sys.exit(f"Not found: {hdf5_path}")

    out_dir = Path(args.out) if args.out else hdf5_path.with_suffix("").with_name(
        hdf5_path.stem + "_viz"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    flip_v = not args.no_flip

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if not demos:
            sys.exit("No demo_* in data/")
        title = "" if args.no_title else task_title(f)

        if args.all:
            targets = demos
        else:
            if args.demo >= len(demos):
                sys.exit(f"--demo {args.demo} >= {len(demos)} demos")
            targets = [f"demo_{args.demo}"]

        # Detect keys once on first demo
        ag_k, eh_k = detect_obs_keys(f["data"][targets[0]]["obs"])
        print(f"obs keys: {ag_k}, {eh_k}   flip_v={flip_v}   fps={args.fps}")
        if title:
            print(f"task: {title}")

        for dk in targets:
            obs = f["data"][dk]["obs"]
            ag = np.asarray(obs[ag_k])
            eh = np.asarray(obs[eh_k])
            stacked = stack_frames(ag, eh, flip_v=flip_v)
            if title:
                stacked = np.stack([annotate(fr, title) for fr in stacked])
            out_path = out_dir / f"{dk}.mp4"
            write_mp4(stacked, out_path, fps=args.fps)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  {dk}: {len(stacked)} frames -> {out_path}  ({size_mb:.2f} MB)")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
