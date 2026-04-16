#!/usr/bin/env python3
"""
Inspect LIBERO robomimic-style HDF5 demonstration files.

Prints group/dataset hierarchy, shapes, dtypes, and important attributes
(env_args, problem_info, etc.) to help design a Diffusion Policy adapter.

Examples:
  python inspect_hdf5.py
  python inspect_hdf5.py --path libero/datasets/libero_spatial
  python inspect_hdf5.py --path libero/datasets/libero_spatial/some_task_demo.hdf5 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    import h5py
except ImportError as e:
    raise SystemExit(
        "h5py is required. Install with: pip install h5py\n"
        f"Original error: {e}"
    ) from e
import numpy as np


def _decode_attr(val: Any) -> Any:
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return val
    if isinstance(val, np.ndarray) and val.dtype.kind in ("S", "O"):
        try:
            return val.astype(str).tolist()
        except Exception:
            return val
    return val


def iter_hdf5_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix.lower() in (".hdf5", ".h5"):
            yield path
        else:
            raise ValueError(f"Not an HDF5 file: {path}")
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".hdf5", ".h5"):
                yield p
    else:
        raise FileNotFoundError(path)


def print_attrs(indent: str, obj: h5py.Group | h5py.Dataset) -> None:
    if not obj.attrs:
        return
    print(f"{indent}Attributes:")
    for k in sorted(obj.attrs.keys()):
        v = obj.attrs[k]
        decoded = _decode_attr(v)
        if isinstance(decoded, str) and len(decoded) > 500:
            preview = decoded[:500] + " ... [truncated]"
        else:
            preview = decoded
        print(f"{indent}  {k}: {preview!r}")


def describe_dataset(indent: str, ds: h5py.Dataset) -> None:
    print(f"{indent}[Dataset] shape={tuple(ds.shape)} dtype={ds.dtype}")


def walk_group(
    prefix: str,
    grp: h5py.Group,
    indent: str,
    max_depth: int,
    depth: int,
) -> None:
    if depth > max_depth:
        print(f"{indent}... (max_depth={max_depth} reached)")
        return
    for key in sorted(grp.keys()):
        path = f"{prefix}/{key}" if prefix else key
        obj = grp[key]
        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {path}/")
            print_attrs(indent + "  ", obj)
            walk_group(path, obj, indent + "  ", max_depth, depth + 1)
        else:
            print(f"{indent}[Dataset] {path}")
            describe_dataset(indent + "  ", obj)
            print_attrs(indent + "  ", obj)


def inspect_file(
    filepath: Path,
    *,
    all_demos: bool,
    max_demos: int,
    full_tree: bool,
    tree_max_depth: int,
    verbose: bool,
) -> None:
    print("=" * 80)
    print(f"File: {filepath.resolve()}")
    print(f"Size: {filepath.stat().st_size / (1024**2):.2f} MB")
    print("=" * 80)

    with h5py.File(filepath, "r") as f:
        print("\n--- Root ---")
        print_attrs("", f)

        if "data" not in f:
            print("\n[Warning] No top-level 'data' group (unexpected for LIBERO demos).")
            if full_tree:
                walk_group("", f, "", tree_max_depth, 0)
            return

        data = f["data"]
        print("\n--- Group: data ---")
        print_attrs("  ", data)

        # Pretty-print JSON-like attrs if possible
        for attr_name in ("env_args", "problem_info"):
            if attr_name in data.attrs:
                raw = data.attrs[attr_name]
                s = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                try:
                    parsed = json.loads(s)
                    print(f"\n--- Parsed data.attrs['{attr_name}'] (JSON) ---")
                    print(json.dumps(parsed, indent=2)[:8000])
                    if len(json.dumps(parsed)) > 8000:
                        print("... [truncated]")
                except json.JSONDecodeError:
                    if verbose:
                        print(f"\n--- data.attrs['{attr_name}'] (raw) ---\n{s[:2000]}")

        demo_keys = [k for k in data.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)

        if "mask" in f:
            print("\n--- Group: mask (robomimic filter keys) ---")
            mask_grp = f["mask"]
            for mk in sorted(mask_grp.keys()):
                d = mask_grp[mk]
                if isinstance(d, h5py.Dataset):
                    arr = d[()]
                    n = len(arr) if hasattr(arr, "__len__") else "?"
                    print(f"  mask/{mk}: shape={getattr(d, 'shape', None)} dtype={d.dtype} (n={n})")

        if not demo_keys:
            print("\n[Warning] No demo_* groups under data/")
            return

        shown_cap = len(demo_keys) if all_demos else min(max_demos, len(demo_keys))
        print(f"\n--- Demos: {len(demo_keys)} total (showing {shown_cap}) ---")

        to_show = demo_keys if all_demos else demo_keys[:max_demos]

        for dk in to_show:
            g = data[dk]
            print(f"\n  === {dk} ===")
            if isinstance(g, h5py.Dataset):
                describe_dataset("    ", g)
                continue
            print_attrs("    ", g)
            for sub in sorted(g.keys()):
                subpath = f"data/{dk}/{sub}"
                child = g[sub]
                if isinstance(child, h5py.Group):
                    print(f"\n    [Group] {sub}/")
                    print_attrs("      ", child)
                    if sub == "obs":
                        for ok in sorted(child.keys()):
                            ds = child[ok]
                            if isinstance(ds, h5py.Dataset):
                                print(
                                    f"      obs/{ok}: shape={tuple(ds.shape)} dtype={ds.dtype}"
                                )
                            else:
                                print(f"      obs/{ok}: [nested Group]")
                                walk_group(
                                    f"{subpath}/obs/{ok}",
                                    ds,
                                    "        ",
                                    tree_max_depth,
                                    0,
                                )
                    elif full_tree:
                        walk_group(subpath, child, "      ", tree_max_depth, 0)
                else:
                    describe_dataset(f"    ", child)
                    print(f"    {sub}: shape={tuple(child.shape)} dtype={child.dtype}")

        if not all_demos and len(demo_keys) > len(to_show):
            print(
                f"\n  ... ({len(demo_keys) - len(to_show)} more demos not shown; "
                "use --all-demos or increase --max-demos)"
            )

        if full_tree:
            print("\n--- Full tree (first demo only) ---")
            first = data[demo_keys[0]]
            if isinstance(first, h5py.Group):
                walk_group(f"data/{demo_keys[0]}", first, "", tree_max_depth, 0)


def default_spatial_dir() -> Path:
    """libero/datasets/libero_spatial relative to this script's parent (LIBERO repo root)."""
    here = Path(__file__).resolve().parent
    return here / "libero" / "datasets" / "libero_spatial"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LIBERO HDF5 demonstration files.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="HDF5 file or directory containing .hdf5 files (default: libero/datasets/libero_spatial next to this script)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="When scanning a directory, inspect at most this many files (default: 5). Use 0 for all.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=2,
        help="Max number of demo_* groups to print per file (default: 2).",
    )
    parser.add_argument(
        "--all-demos",
        action="store_true",
        help="Print all demos (still capped by --max-demos unless you raise it).",
    )
    parser.add_argument(
        "--full-tree",
        action="store_true",
        help="Also dump a recursive tree for obs and first demo.",
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=6,
        help="Max recursion depth for --full-tree (default: 6).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print long raw attrs if JSON parse fails.",
    )
    args = parser.parse_args()

    base = Path(args.path) if args.path else default_spatial_dir()
    if not base.exists():
        print(f"Path not found: {base}", file=sys.stderr)
        print("Pass --path to your libero_spatial folder or a single .hdf5 file.", file=sys.stderr)
        sys.exit(1)

    files = list(iter_hdf5_files(base))
    if not files:
        print(f"No .hdf5 files under {base}", file=sys.stderr)
        sys.exit(1)

    max_files = args.max_files if args.max_files > 0 else len(files)
    for i, fp in enumerate(files[:max_files]):
        inspect_file(
            fp,
            all_demos=args.all_demos,
            max_demos=args.max_demos,
            full_tree=args.full_tree,
            tree_max_depth=args.tree_max_depth,
            verbose=args.verbose,
        )
        if i < min(len(files), max_files) - 1:
            print("\n")

    if len(files) > max_files:
        print(f"\n... ({len(files) - max_files} more files not shown; increase --max-files or pass a single file)")


if __name__ == "__main__":
    main()
