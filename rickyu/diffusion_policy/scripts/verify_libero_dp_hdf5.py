#!/usr/bin/env python3
"""
Verify a converted LIBERO HDF5 matches libero_spatial_one shape_meta (no PyTorch).
Run from diffusion_policy repo root: python scripts/verify_libero_dp_hdf5.py
"""

import json
import sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HDF5 = ROOT / "data" / "libero" / "spatial_one_dp_demo.hdf5"

EXPECTED_OBS = {
    "agentview_image": ("rgb", (128, 128, 3)),
    "robot0_eye_in_hand_image": ("rgb", (128, 128, 3)),
    "robot0_joint_qpos": ("low_dim", (7,)),
    "robot0_gripper_qpos": ("low_dim", (2,)),
}


def main() -> None:
    hdf5_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HDF5
    if not hdf5_path.is_file():
        print(f"Missing file: {hdf5_path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        assert demos, "no demo_* in data/"
        d0 = f["data"][demos[0]]
        obs = d0["obs"]
        for k, (_, exp_hwc) in EXPECTED_OBS.items():
            assert k in obs, f"missing obs/{k}"
            shp = obs[k].shape
            if exp_hwc[-1] == 3:  # image HWC
                assert shp[1:] == exp_hwc, f"{k} bad shape {shp} expected (*,{exp_hwc})"
            else:
                assert shp[1:] == exp_hwc, f"{k} bad shape {shp}"
        a = d0["actions"].shape
        assert a[-1] == 7, a

        pi = f["data"].attrs.get("problem_info")
        if pi is not None:
            if isinstance(pi, bytes):
                pi = pi.decode("utf-8")
            info = json.loads(pi)
            print("problem_info language:", info.get("language_instruction", "")[:80])

    print(f"OK: {hdf5_path}")
    print(f"  demos={len(demos)}, checked {demos[0]} obs keys + actions")


if __name__ == "__main__":
    main()
