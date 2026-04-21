#!/usr/bin/env python3
"""
【内存优化版】rdp_pkl_to_zarr.py
采用流式写入 (Append Mode)，避免一次性加载所有数据导致 OOM (Killed)。
支持进度条显示。
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import shutil

import numpy as np
import zarr
import numcodecs
from tqdm import tqdm  # 需要 pip install tqdm

# 保证能从项目根 import reactive_diffusion_policy
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from reactive_diffusion_policy.model.common.pca_embedding import PCAEmbedding

# Key 定义
KEY_CAMERA_F = "camera_f"
KEY_CAMERA_R = "camera_r"
KEY_TACTILE_IMAGE = "Tactile_Image_Right"
KEY_MARKER_DXDY = "Marker_Tracking_Right_DXDY"
KEY_MARKER_DXDY_EMB = "Marker_Tracking_Right_DXDY_emb"
KEY_JOINT_STATES = "joint_states_single"
KEY_JOINT_CTRL = "joint_ctrl_single"
KEY_ACTION = "action"
KEY_TIMESTAMP = "timestamp"


def _ensure_1d_float32(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).ravel()


def load_pkl_frames(pkl_path: str) -> list:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return getattr(data, "frames", data) if not isinstance(data, list) else data


def init_zarr_structure(store, first_sample: dict, pca_dim: int = 15, use_pca: bool = False):
    """根据第一个样本的形状初始化 Zarr 数组结构（Chunked, Resizeable）"""
    root = zarr.group(store=store)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    # 压缩器
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

    # 1. Meta: episode_ends (可变长)
    meta_group.empty("episode_ends", shape=(0,), chunks=(1000,), dtype="int64", compressor=compressor)

    # 2. Data Keys 推断
    # 图像: (T, C, H, W) -> chunks=(1, C, H, W)
    # 向量: (T, D) -> chunks=(10000, D)
    
    def create_resizable_array(key, sample_arr):
        # sample_arr 是单帧数据，shape=(...), 存入 zarr 时 shape=(0, ...)
        shape = (0,) + sample_arr.shape
        if sample_arr.ndim >= 2: # 图像 (C, H, W)
             chunks = (1,) + sample_arr.shape
        else: # 向量 (D,)
             chunks = (10000,) + sample_arr.shape
        
        data_group.empty(key, shape=shape, chunks=chunks, dtype=sample_arr.dtype, compressor=compressor)

    # 遍历 sample 创建 dataset
    create_resizable_array(KEY_TIMESTAMP, np.float32(0.0)) # scalar
    create_resizable_array(KEY_CAMERA_F, first_sample[KEY_CAMERA_F])
    create_resizable_array(KEY_CAMERA_R, first_sample[KEY_CAMERA_R])
    create_resizable_array(KEY_TACTILE_IMAGE, first_sample[KEY_TACTILE_IMAGE])
    
    # Marker
    marker = first_sample[KEY_MARKER_DXDY] # (63, 2)
    marker_flat = marker.reshape(-1).astype(np.float32) # (126,)
    create_resizable_array(KEY_MARKER_DXDY, marker_flat)
    
    if use_pca:
        # PCA embedding 是 (pca_dim,)
        emb_sample = np.zeros((pca_dim,), dtype=np.float32)
        create_resizable_array(KEY_MARKER_DXDY_EMB, emb_sample)

    # Joint & Action
    js = _ensure_1d_float32(first_sample[KEY_JOINT_STATES])
    create_resizable_array(KEY_JOINT_STATES, js)
    
    jc = _ensure_1d_float32(first_sample[KEY_JOINT_CTRL])
    # Action 维度同 Joint Ctrl
    create_resizable_array(KEY_ACTION, jc)
    
    return root


def process_and_append_episode(root, frames, pca_model=None):
    """处理单个 episode (pkl) 并追加到 Zarr"""
    if not frames:
        return 0

    n = len(frames)
    
    # 1. 提取并预处理本 Episode 的数据
    ts = np.array([f["timestamp"] for f in frames], dtype=np.float32)
    camera_f = np.stack([f[KEY_CAMERA_F] for f in frames], axis=0)
    camera_r = np.stack([f[KEY_CAMERA_R] for f in frames], axis=0)
    tactile = np.stack([f[KEY_TACTILE_IMAGE] for f in frames], axis=0)
    
    marker = np.stack([f[KEY_MARKER_DXDY] for f in frames], axis=0)
    marker_flat = marker.reshape(n, -1).astype(np.float32)

    joint_states = np.stack([_ensure_1d_float32(f[KEY_JOINT_STATES]) for f in frames], axis=0)
    joint_ctrl = np.stack([_ensure_1d_float32(f[KEY_JOINT_CTRL]) for f in frames], axis=0)

    # 2. 计算 Action (Shifted Joint Ctrl)
    # Action[t] = Joint_Ctrl[t+1]
    # 最后一个动作重复倒数第二个 (copy padding)
    action = np.zeros_like(joint_ctrl, dtype=np.float32)
    action[:-1] = joint_ctrl[1:]
    if n > 1:
        action[-1] = joint_ctrl[-2] # 保持静止或维持趋势
    else:
        action[-1] = joint_ctrl[-1]

    # 3. PCA (如有)
    marker_emb = None
    if pca_model is not None:
        # pca_reduction 期望输入 (N, 126)
        marker_emb = pca_model.pca_reduction(marker_flat).astype(np.float32)

    # 4. 追加写入 Zarr (Append)
    dg = root["data"]
    dg[KEY_TIMESTAMP].append(ts)
    dg[KEY_CAMERA_F].append(camera_f)
    dg[KEY_CAMERA_R].append(camera_r)
    dg[KEY_TACTILE_IMAGE].append(tactile)
    dg[KEY_MARKER_DXDY].append(marker_flat)
    if marker_emb is not None:
        dg[KEY_MARKER_DXDY_EMB].append(marker_emb)
    dg[KEY_JOINT_STATES].append(joint_states)
    dg[KEY_ACTION].append(action)

    return n


def run_streaming(pkl_paths: list, out_zarr_path: str, pca_dir: str | None = None, pca_dim: int = 15):
    # 0. 准备 PCA
    pca_model = None
    if pca_dir:
        pca_dir = os.path.expanduser(pca_dir)
        transform_path = os.path.join(pca_dir, "pca_transform_matrix.npy")
        mean_path = os.path.join(pca_dir, "pca_mean_matrix.npy")
        if not os.path.isfile(transform_path):
             raise FileNotFoundError(f"Missing PCA matrix: {transform_path}")
        print(f"Loaded PCA from {pca_dir}")
        pca_model = PCAEmbedding(
            n_components=pca_dim, normalize=False, mode="Eval", store=False,
            transformation_matrix_path=transform_path, mean_matrix_path=mean_path
        )

    # 1. 准备 Zarr Store
    if os.path.exists(out_zarr_path):
        print(f"Removing existing zarr: {out_zarr_path}")
        shutil.rmtree(out_zarr_path)
    
    store = zarr.DirectoryStore(out_zarr_path)
    
    # 2. 读取第一个 PKL 以初始化 Zarr 结构
    print("Reading first pickle to initialize Zarr structure...")
    first_frames = load_pkl_frames(pkl_paths[0])
    if not first_frames:
        raise RuntimeError(f"First pickle {pkl_paths[0]} is empty!")
    
    root = init_zarr_structure(store, first_frames[0], pca_dim=pca_dim, use_pca=(pca_model is not None))
    
    # 3. 流式处理 Loop
    total_steps = 0
    episode_ends = []
    
    print(f"Start converting {len(pkl_paths)} episodes...")
    for pkl_path in tqdm(pkl_paths, desc="Processing"):
        frames = load_pkl_frames(pkl_path)
        steps = process_and_append_episode(root, frames, pca_model)
        
        if steps > 0:
            total_steps += steps
            episode_ends.append(total_steps)
            
    # 4. 写入 episode_ends
    root["meta"]["episode_ends"].append(np.array(episode_ends, dtype=np.int64))
    
    print(f"\n✅ Done! Saved to: {out_zarr_path}")
    print(f"Total Episodes: {len(episode_ends)}")
    print(f"Total Steps: {total_steps}")


def main():
    parser = argparse.ArgumentParser(description="流式转换 PKL 到 Zarr (低内存占用)")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--pkl-dir", type=str)
    g.add_argument("--pkl-files", type=str, nargs="+")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--pca-dir", type=str, default=None)
    parser.add_argument("--pca-dim", type=int, default=15)
    
    args = parser.parse_args()
    
    if args.pkl_dir:
        pkl_dir = os.path.expanduser(args.pkl_dir)
        pkl_paths = [os.path.join(pkl_dir, f) for f in sorted(os.listdir(pkl_dir)) if f.endswith(".pkl")]
    else:
        pkl_paths = sorted([os.path.expanduser(p) for p in args.pkl_files])

    if not pkl_paths:
        print("No pkl files found.")
        return

    run_streaming(pkl_paths, os.path.expanduser(args.output), args.pca_dir, args.pca_dim)

if __name__ == "__main__":
    main()