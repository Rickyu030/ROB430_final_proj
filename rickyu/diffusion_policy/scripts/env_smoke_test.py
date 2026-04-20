#!/usr/bin/env python3
"""
Layer-4 prerequisite smoke test.

Verifies that:
1. MuJoCo can initialize a rendering backend (EGL preferred, OSMesa fallback)
2. libero.libero.envs.OffScreenRenderEnv constructs from the HDF5 env_args
3. env.reset() + env.set_init_state(states[0]) yield obs with expected keys/shapes
4. env.step([0]*7) runs one sim step and returns (obs, reward, done, info)

Usage:
    python scripts/env_smoke_test.py [--hdf5 PATH] [--demo IDX] [--backend egl|osmesa]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

LIBERO_PKG_ROOT = Path("/home/rickyu/DP/LIBERO/libero/libero")


def resolve_bddl(env_args: dict, hdf5_path: Path) -> str | None:
    """Return an existing BDDL file path using three strategies."""
    # 1) env_args.bddl_file / env_kwargs.bddl_file_name (already-patched)
    candidates = [env_args.get("bddl_file")]
    ekw = env_args.get("env_kwargs") or {}
    candidates.extend([ekw.get("bddl_file_name"), ekw.get("bddl_file")])
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    # 2) "bddl_files/..." suffix match on local LIBERO tree
    for c in candidates:
        if c and "bddl_files/" in c:
            rel = c.split("bddl_files/", 1)[1]
            p = LIBERO_PKG_ROOT / "bddl_files" / rel
            if p.is_file():
                return str(p)
    # 3) Fallback: derive from HDF5 filename (strip _demo.hdf5 / _dp_demo.hdf5)
    stem = hdf5_path.name
    for suf in ("_dp_demo.hdf5", "_demo.hdf5", ".hdf5"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    # Search all LIBERO subdirs for matching .bddl
    for bddl in (LIBERO_PKG_ROOT / "bddl_files").rglob(f"{stem}.bddl"):
        return str(bddl)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="data/libero/spatial_one_dp_demo.hdf5")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--backend", choices=["egl", "osmesa", "glfw"], default="egl")
    ap.add_argument("--egl-device", default="0")
    ap.add_argument(
        "--bddl", default=None,
        help="Explicit .bddl path override (use when HDF5 env_args has stale path).",
    )
    args = ap.parse_args()

    # MUST set before importing mujoco / robosuite / libero
    os.environ["MUJOCO_GL"] = args.backend
    if args.backend == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", args.egl_device)
    print(f"[env] MUJOCO_GL={os.environ.get('MUJOCO_GL')}  "
          f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')}")

    import numpy as np
    import h5py
    import mujoco
    import robosuite
    print(f"[versions] mujoco={mujoco.__version__}  robosuite={robosuite.__version__}")

    # Probe EGL/OSMesa by creating a tiny GLContext.
    try:
        from mujoco import GLContext
        ctx = GLContext(64, 64)
        ctx.make_current()
        print(f"[gl] {args.backend} GLContext OK")
        ctx.free()
    except Exception as e:
        print(f"[gl] {args.backend} GLContext FAILED: {type(e).__name__}: {e}")
        sys.exit(3)

    # Now import LIBERO (imports robosuite under the hood)
    from libero.libero.envs import OffScreenRenderEnv

    hdf5 = Path(args.hdf5).resolve()
    if not hdf5.is_file():
        sys.exit(f"hdf5 not found: {hdf5}")
    print(f"[data] {hdf5}")

    with h5py.File(hdf5, "r") as f:
        ea = json.loads(f["data"].attrs["env_args"])
        demo_key = f"demo_{args.demo}"
        if demo_key not in f["data"]:
            sys.exit(f"{demo_key} not in {hdf5}")
        if "states" not in f["data"][demo_key]:
            sys.exit(f"no /data/{demo_key}/states in {hdf5}")
        init_state = np.asarray(f["data"][demo_key]["states"][0])
        first_gt_img = np.asarray(f["data"][demo_key]["obs"]["agentview_image"][0])

    if args.bddl:
        if not os.path.isfile(args.bddl):
            sys.exit(f"--bddl file not found: {args.bddl}")
        bddl = args.bddl
    else:
        bddl = resolve_bddl(ea, hdf5)
    if not bddl:
        sys.exit(
            f"bddl not resolvable. Pass --bddl explicitly.\n"
            f"  env_args.bddl_file: {ea.get('bddl_file')}\n"
            f"  hdf5 filename:      {hdf5.name}"
        )
    print(f"[bddl] {bddl}")
    print(f"[init_state] shape={init_state.shape}  dtype={init_state.dtype}")

    # Minimal invocation: only pass bddl + camera dims (plan step 1 gotcha #4)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=0 if args.backend == "egl" else -1,
    )
    print(f"[env] language_instruction: {env.language_instruction}")

    env.seed(0)
    env.reset()
    obs = env.set_init_state(init_state)

    print(f"[obs] keys containing image/joint/gripper/eef:")
    for k in sorted(obs.keys()):
        if any(s in k for s in ("image", "joint", "gripper", "eef")):
            v = obs[k]
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", None)
            print(f"       {k:34s} shape={shape} dtype={dtype}")

    required = [
        "agentview_image", "robot0_eye_in_hand_image",
        "robot0_joint_pos", "robot0_gripper_qpos",
    ]
    missing = [k for k in required if k not in obs]
    if missing:
        sys.exit(f"[FAIL] missing obs keys: {missing}")

    # Shape sanity
    assert obs["agentview_image"].shape == (128, 128, 3), obs["agentview_image"].shape
    assert obs["robot0_eye_in_hand_image"].shape == (128, 128, 3)

    # Compare first frame of env vs HDF5 (informational — they may differ)
    ag = obs["agentview_image"]
    diff = np.abs(ag.astype(np.int32) - first_gt_img.astype(np.int32))
    print(f"[img cmp] env first frame vs HDF5 first frame:")
    print(f"       L1 mean  = {diff.mean():.2f}")
    print(f"       L1 max   = {diff.max()}")
    print(f"       identical = {np.array_equal(ag, first_gt_img)}")

    # Step once with zero action
    obs2, reward, done, info = env.step([0.0] * 7)
    print(f"[step] reward={reward}  done={done}  info_keys={list(info.keys())}")

    env.close()
    print("\n[OK] Smoke test passed.")


if __name__ == "__main__":
    main()
