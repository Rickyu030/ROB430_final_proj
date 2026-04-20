#!/usr/bin/env python3
"""
Layer-4 evaluation: run trained policy in LIBERO MuJoCo simulator.

For each selected demo i:
    env.reset()
    obs = env.set_init_state(demo_states[i][0])          # demo's starting state
    history = [obs, obs]                                  # pad to n_obs_steps
    while not done and steps < max_steps:
        action_chunk = policy.predict_action(history)["action"][0]  # (n_action_steps, 7)
        for a in action_chunk:
            obs, r, done, info = env.step(a)
            history.append(obs)
            write frame to mp4
            if done: success; break

Writes:
    <out>/rollouts_<ts>/demo_<i>_<OK|FAIL>.mp4
    <out>/rollouts_<ts>/rollout_summary.png
    <out>/rollouts_<ts>/results.json

Usage:
    python scripts/eval_libero_rollout.py \
        --checkpoint data/outputs/.../latest.ckpt \
        --hdf5       data/libero/spatial_one_dp_demo.hdf5 \
        --bddl       /home/rickyu/DP/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl \
        --demos 0,1 \
        --max-steps 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# CRITICAL: set MUJOCO_GL before any mujoco/robosuite import
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

LIBERO_PKG_ROOT = Path("/home/rickyu/DP/LIBERO/libero/libero")

# Mapping from live env obs keys -> policy-expected obs keys (see shape_meta in
# /home/rickyu/DP/diffusion_policy/diffusion_policy/config/task/libero_spatial_one.yaml).
OBS_RENAME = {
    "robot0_joint_pos": "robot0_joint_qpos",
}


def resolve_bddl(env_args: dict, hdf5_path: Path, override: str | None) -> str | None:
    if override:
        return override if os.path.isfile(override) else None
    candidates = [env_args.get("bddl_file")]
    ekw = env_args.get("env_kwargs") or {}
    candidates.extend([ekw.get("bddl_file_name"), ekw.get("bddl_file")])
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    for c in candidates:
        if c and "bddl_files/" in c:
            rel = c.split("bddl_files/", 1)[1]
            p = LIBERO_PKG_ROOT / "bddl_files" / rel
            if p.is_file():
                return str(p)
    return None


def build_obs_tensor(hist: deque, rgb_keys, low_keys, rename_map, device):
    """history deque (len = n_obs_steps) -> dict of (1, To, ...) tensors on device."""
    import numpy as np
    import torch

    out = {}
    for k in rgb_keys:
        imgs = np.stack([o[k] for o in hist], axis=0)               # (To, H, W, 3) uint8
        imgs = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0   # (To, 3, H, W)
        out[k] = torch.from_numpy(imgs).unsqueeze(0).to(device)      # (1, To, 3, H, W)
    for k in low_keys:
        src_k = next((sk for sk, dk in rename_map.items() if dk == k), k)
        lookup_k = src_k if src_k in hist[0] else k
        arr = np.stack([o[lookup_k] for o in hist], axis=0).astype(np.float32)
        out[k] = torch.from_numpy(arr).unsqueeze(0).to(device)
    return out


def put_text(img, text, pos=(6, 16), color=(255, 255, 255), scale=0.45):
    try:
        import cv2
        img2 = img.copy()
        cv2.putText(img2, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img2, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)
        return img2
    except Exception:
        return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--bddl", default=None,
                    help="Explicit .bddl path (use when HDF5 env_args is stale)")
    ap.add_argument("--demos", default="0",
                    help="Comma list of demo indices, or 'range(a,b)'")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default=None,
                    help="Output dir (default: data/libero/spatial_one_dp_demo_viz/rollouts_<ts>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-flip", action="store_true",
                    help="Don't flip agentview vertically when writing mp4")
    args = ap.parse_args()

    import numpy as np
    import torch
    import dill
    import hydra
    import h5py
    import imageio.v2 as imageio
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # ---- Parse --demos ----
    demos_arg = args.demos.strip()
    if demos_arg.startswith("range("):
        a, b = [int(x) for x in demos_arg[len("range("):-1].split(",")]
        demo_idxs = list(range(a, b))
    else:
        demo_idxs = [int(x) for x in demos_arg.split(",") if x.strip()]
    print(f"[args] demos={demo_idxs}  max_steps={args.max_steps}")

    # ---- Load policy ----
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        sys.exit(f"ckpt not found: {ckpt}")
    print(f"[load] {ckpt}")
    t0 = time.time()
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    ws = cls(cfg, output_dir="/tmp/dp_eval_layer4")
    ws.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    device = torch.device(args.device)
    policy.to(device).eval()
    print(f"[load] done in {time.time()-t0:.1f}s  "
          f"horizon={cfg.horizon}  n_obs={cfg.n_obs_steps}  n_act={cfg.n_action_steps}")

    n_obs = cfg.n_obs_steps
    n_act = cfg.n_action_steps
    shape_meta = cfg.task.shape_meta
    rgb_keys, low_keys = [], []
    for k, v in shape_meta["obs"].items():
        (rgb_keys if v.get("type", "low_dim") == "rgb" else low_keys).append(k)
    print(f"[obs] rgb={rgb_keys}  low={low_keys}")

    # ---- Read HDF5 demos ----
    hdf5 = Path(args.hdf5).resolve()
    with h5py.File(hdf5, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo_states = {}
        demo_lens = {}
        for i in demo_idxs:
            dk = f"demo_{i}"
            if dk not in f["data"]:
                sys.exit(f"{dk} not in {hdf5}")
            demo_states[i] = np.asarray(f["data"][dk]["states"][0])
            demo_lens[i] = int(f["data"][dk]["actions"].shape[0])
    bddl = resolve_bddl(env_args, hdf5, args.bddl)
    if not bddl:
        sys.exit(f"bddl unresolved. Pass --bddl.")
    print(f"[bddl] {bddl}")

    # ---- Build env once (same task for all demos) ----
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=0,
    )
    print(f"[env] language: {env.language_instruction}")

    # ---- Output dir ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (
        hdf5.with_suffix("").with_name(hdf5.stem + "_viz") / f"rollouts_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    # ---- Rollout loop ----
    results = []
    t_eval_start = time.time()

    for idx, demo_i in enumerate(demo_idxs):
        print(f"\n=== rollout {idx+1}/{len(demo_idxs)}  demo_{demo_i} "
              f"(demo_len={demo_lens[demo_i]}) ===")

        env.seed(demo_i)
        env.reset()
        obs = env.set_init_state(demo_states[demo_i])

        hist = deque([obs] * n_obs, maxlen=n_obs)
        frames = []
        success = False
        step = 0
        last_reward = 0.0

        def _append_frame(o, s, succ, rew):
            ag = o["agentview_image"]
            if not args.no_flip:
                ag = ag[::-1]
            ag = np.ascontiguousarray(ag)
            ag = put_text(ag, f"demo_{demo_i}  t={s:3d}/{args.max_steps}")
            ag = put_text(ag, f"r={rew:.1f}  success={int(succ)}", pos=(6, 32),
                          color=(0, 255, 0) if succ else (255, 255, 255))
            frames.append(ag)

        _append_frame(obs, 0, False, 0.0)

        t_rollout = time.time()
        while step < args.max_steps and not success:
            obs_dict = build_obs_tensor(hist, rgb_keys, low_keys, OBS_RENAME, device)
            with torch.no_grad():
                action_chunk = policy.predict_action(obs_dict)["action"][0].cpu().numpy()

            for a in action_chunk:
                obs, reward, done, info = env.step(a)
                hist.append(obs)
                last_reward = float(reward)
                step += 1
                _append_frame(obs, step, bool(done), last_reward)
                if done:
                    success = True
                    break
                if step >= args.max_steps:
                    break

            if (step % 40 == 0) and not success:
                elapsed = time.time() - t_rollout
                print(f"  step={step:4d}  reward={last_reward:.2f}  "
                      f"({elapsed:.1f}s, {elapsed/step:.3f}s/step)")

        elapsed = time.time() - t_rollout
        tag = "OK" if success else "FAIL"
        mp4_path = out_dir / f"demo_{demo_i}_{tag}.mp4"
        writer = imageio.get_writer(str(mp4_path), fps=args.fps, codec="libx264",
                                    quality=8, macro_block_size=1)
        for fr in frames:
            writer.append_data(fr)
        writer.close()
        size_mb = mp4_path.stat().st_size / (1024 * 1024)
        print(f"  -> {tag}  steps={step}  wrote {mp4_path.name} ({size_mb:.2f} MB)"
              f"  rollout_time={elapsed:.1f}s")
        results.append(dict(
            demo=demo_i, success=success, steps=step,
            rollout_time_s=round(elapsed, 2), mp4=str(mp4_path),
        ))

    env.close()
    total_elapsed = time.time() - t_eval_start

    # ---- Summary JSON ----
    n = len(results)
    n_ok = sum(r["success"] for r in results)
    avg_steps_succ = (np.mean([r["steps"] for r in results if r["success"]])
                      if n_ok > 0 else None)
    summary = dict(
        checkpoint=str(ckpt),
        hdf5=str(hdf5),
        bddl=bddl,
        n_rollouts=n,
        n_success=n_ok,
        success_rate=n_ok / n if n > 0 else 0.0,
        avg_steps_success=(float(avg_steps_succ) if avg_steps_succ is not None else None),
        total_elapsed_s=round(total_elapsed, 1),
        max_steps=args.max_steps,
        per_demo=results,
    )
    summary_json = out_dir / "results.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] success_rate = {n_ok}/{n} = {summary['success_rate']*100:.1f}%"
          f"  total={total_elapsed:.1f}s")
    print(f"[summary] wrote {summary_json}")

    # ---- Summary plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n + 3), 4.5))
    demo_labels = [f"d{r['demo']}" for r in results]
    steps_list = [r["steps"] for r in results]
    colors = ["tab:green" if r["success"] else "tab:red" for r in results]
    bars = ax.bar(demo_labels, steps_list, color=colors)
    ax.axhline(args.max_steps, color="gray", ls="--", lw=1, alpha=0.6,
               label=f"max_steps={args.max_steps}")
    if avg_steps_succ is not None:
        ax.axhline(avg_steps_succ, color="tab:green", ls=":", lw=1.2, alpha=0.8,
                   label=f"avg success steps={avg_steps_succ:.1f}")
    for b, r in zip(bars, results):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                "OK" if r["success"] else "FAIL",
                ha="center", fontsize=9,
                color="tab:green" if r["success"] else "tab:red")
    ax.set_ylabel("steps taken")
    ax.set_title(
        f"LIBERO rollout  -  success {n_ok}/{n} ({summary['success_rate']*100:.1f}%)\n"
        f"ckpt: {ckpt.parent.parent.name}"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    summary_png = out_dir / "rollout_summary.png"
    fig.savefig(summary_png, dpi=130)
    print(f"[summary] wrote {summary_png}")


if __name__ == "__main__":
    main()
