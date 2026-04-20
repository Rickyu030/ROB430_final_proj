#!/usr/bin/env python3
"""
Convert LIBERO robomimic-style demonstration HDF5 to a layout compatible with
Diffusion Policy's RobomimicReplayImageDataset (obs key names and suffix rules).

Renames:
  agentview_rgb          -> agentview_image
  eye_in_hand_rgb        -> robot0_eye_in_hand_image
  joint_states           -> robot0_joint_qpos   (low_dim normalizer requires *qpos)
  gripper_states         -> robot0_gripper_qpos

Optionally rewrites data.attrs env_args JSON so bddl_file / bddl_file_name point
to this machine's LIBERO checkout (for later robomimic-style env reload).

Examples:
  python convert_libero_hdf5_for_dp.py -i libero/datasets/libero_spatial/foo_demo.hdf5 \\
      -o ../diffusion_policy/data/libero/foo_dp.hdf5

  python convert_libero_hdf5_for_dp.py -i libero/datasets/libero_spatial \\
      -o ../diffusion_policy/data/libero --batch
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import h5py
except ImportError as e:
    raise SystemExit("h5py is required: pip install h5py") from e

# LIBERO original key -> DP / robomimic-style key (must match shape_meta in task yaml)
OBS_KEY_MAP = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "joint_states": "robot0_joint_qpos",
    "gripper_states": "robot0_gripper_qpos",
}


def _repo_libero_package_root() -> Path:
    """.../LIBERO/libero/libero (contains bddl_files, envs)."""
    return Path(__file__).resolve().parent / "libero" / "libero"


def resolve_bddl_on_disk(stored_path: str, libero_package_root: Path) -> str | None:
    """
    Map a path that may contain bddl_files/... anywhere to
    libero_package_root / bddl_files / <rest>.
    """
    if not stored_path:
        return None
    marker = "bddl_files/"
    if marker in stored_path:
        rel = stored_path.split(marker, 1)[1]
        candidate = (libero_package_root / "bddl_files" / rel).resolve()
        if candidate.is_file():
            return str(candidate)
    # try as relative to package root
    p = Path(stored_path)
    if not p.is_absolute():
        cand = (libero_package_root.parent / stored_path).resolve()
        if cand.is_file():
            return str(cand)
    p2 = Path(stored_path)
    if p2.is_file():
        return str(p2.resolve())
    return None


def patch_env_args_json(raw: str | bytes, libero_package_root: Path) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    env_args = json.loads(raw)

    bddl = env_args.get("bddl_file")
    resolved = resolve_bddl_on_disk(bddl or "", libero_package_root)
    if resolved:
        env_args["bddl_file"] = resolved

    env_kwargs = env_args.get("env_kwargs") or {}
    for k in ("bddl_file_name", "bddl_file"):
        v = env_kwargs.get(k)
        if v:
            r = resolve_bddl_on_disk(v, libero_package_root)
            if r:
                env_kwargs[k] = r
    env_args["env_kwargs"] = env_kwargs

    return json.dumps(env_args, separators=(",", ":"))


def convert_one_file(
    src_path: Path,
    dst_path: Path,
    *,
    libero_package_root: Path,
    patch_attrs: bool,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        sdata = src["data"]
        ddata = dst.create_group("data")

        # Root data attrs
        for k in sdata.attrs.keys():
            val = sdata.attrs[k]
            if patch_attrs and k == "env_args":
                try:
                    ddata.attrs[k] = patch_env_args_json(val, libero_package_root)
                except (json.JSONDecodeError, TypeError):
                    ddata.attrs[k] = val
            elif patch_attrs and k == "bddl_file_name":
                r = resolve_bddl_on_disk(
                    val.decode("utf-8") if isinstance(val, bytes) else str(val),
                    libero_package_root,
                )
                ddata.attrs[k] = r if r else val
            else:
                ddata.attrs[k] = val

        demo_keys = sorted(
            [k for k in sdata.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )

        for dk in demo_keys:
            sg = sdata[dk]
            dg = ddata.create_group(dk)

            for ak in sg.attrs.keys():
                dg.attrs[ak] = sg.attrs[ak]

            # actions, states, rewards, dones, robot_states (same names)
            for name in ("actions", "states", "rewards", "dones", "robot_states"):
                if name in sg:
                    sg.copy(name, dg, name=name)

            if "obs" not in sg:
                continue
            sobs = sg["obs"]
            dobs = dg.create_group("obs")

            for libero_k, dp_k in OBS_KEY_MAP.items():
                if libero_k not in sobs:
                    raise KeyError(
                        f"{src_path}: {dk}/obs missing {libero_k!r} "
                        f"(available: {list(sobs.keys())})"
                    )
                sobs.copy(libero_k, dobs, name=dp_k)

        # mask group (robomimic train/val splits) if present
        if "mask" in src:
            src.copy("mask", dst)

    print(f"Wrote {dst_path} ({dst_path.stat().st_size / (1024**2):.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 demos to DP-compatible obs key names."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input .hdf5 file or directory of *_demo.hdf5 files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file path, or output directory when using --batch",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="If input is a directory, convert every *_demo.hdf5 into output dir",
    )
    parser.add_argument(
        "--no-patch-env-args",
        action="store_true",
        help="Do not rewrite env_args / bddl paths for local LIBERO tree",
    )
    parser.add_argument(
        "--libero-root",
        type=str,
        default=None,
        help="Path to .../libero/libero (default: next to this script)",
    )
    args = parser.parse_args()

    libero_root = Path(args.libero_root) if args.libero_root else _repo_libero_package_root()
    if not libero_root.is_dir():
        print(f"Warning: LIBERO package root not found: {libero_root}", file=sys.stderr)

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    patch_attrs = not args.no_patch_env_args

    if args.batch:
        if not inp.is_dir():
            raise SystemExit("--batch requires input to be a directory")
        out.mkdir(parents=True, exist_ok=True)
        files = sorted(inp.glob("*_demo.hdf5"))
        if not files:
            raise SystemExit(f"No *_demo.hdf5 under {inp}")
        for f in files:
            dst = out / f.name.replace("_demo.hdf5", "_dp_demo.hdf5")
            convert_one_file(f, dst, libero_package_root=libero_root, patch_attrs=patch_attrs)
        print(f"Converted {len(files)} files -> {out}")
        return

    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")
    if out.is_dir() or str(out).endswith("/"):
        out.mkdir(parents=True, exist_ok=True)
        dst = out / inp.name.replace("_demo.hdf5", "_dp_demo.hdf5")
    else:
        dst = out
    convert_one_file(inp, dst, libero_package_root=libero_root, patch_attrs=patch_attrs)


if __name__ == "__main__":
    main()
