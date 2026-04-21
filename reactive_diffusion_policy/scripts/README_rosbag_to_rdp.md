# ROS1 rosbag → pkl → zarr → 训练 数据流程

## 流程概览

1. **rosbag → pkl**：`rosbag_to_rdp_pkl.py`，按原始 topic 名作为 key 存（camera_f、camera_r、Tactile_Image_Right、Marker_Tracking_Right_DXDY、joint_states_single、joint_ctrl_single）。
2. **pkl → zarr**：`rdp_pkl_to_zarr.py`，把多个 pkl（每个视为一个 episode）合并成 RDP 训练用的 replay_buffer.zarr。
3. **训练**：用 `train.py` + 任务 config，其中 `shape_meta` 和 `dataset_path` 与 zarr 里 key 一致。

---

## 1. rosbag → pkl

### 依赖

- ROS1（rosbag、cv_bridge）
- numpy

### 运行

```bash
# 输出默认与 bag 同目录、同名 .pkl
python scripts/rosbag_to_rdp_pkl.py /path/to/your.bag

# 指定输出
python scripts/rosbag_to_rdp_pkl.py /path/to/your.bag -o data/my_session.pkl
```
python scripts/rosbag_to_rdp_pkl.py --bag-dir /home/yfx/turbo/Dataset/peeling/test

### pkl 结构

- `data.frames`：list of dict
- 每帧 dict 的 key：`timestamp`, `camera_f`, `camera_r`, `Tactile_Image_Right`, `Marker_Tracking_Right_DXDY`, `joint_states_single`, `joint_ctrl_single`
- 图像保持 rosbag 原始分辨率；关节为 `{ "position", "velocity", "effort" }` 的 numpy 数组

修改 topic 名或对齐参考：改 `rosbag_to_rdp_pkl.py` 顶部常量。

---

## 2. pkl → zarr

### 作用

把「你的 pkl」转成 RDP 的 `ReplayBuffer` 格式 zarr，key 与 pkl 一致，并生成 `action`（action[t] = 下一帧的 joint_ctrl position）。

### 运行

```bash
# 一个目录下所有 .pkl 合并（按文件名排序，每个 pkl 一个 episode）
python scripts/rdp_pkl_to_zarr.py --pkl-dir data/my_pkls -o data/my_task/replay_buffer.zarr

# 或指定多个 pkl
python scripts/rdp_pkl_to_zarr.py --pkl-files a.pkl b.pkl c.pkl -o data/my_task/replay_buffer.zarr

# 对触觉做 PCA 降维（126→15），写入 Marker_Tracking_Right_DXDY_emb（需有 RDP 的 PCA 矩阵目录）
python scripts/rdp_pkl_to_zarr.py --pkl-dir /home/yfx/turbo/Dataset/peeling/test -o /home/yfx/turbo/Dataset/peeling/test/test.zarr --pca-dir /home/yfx/turbo/reactive_diffusion_policy/data/PCA_Transform_GelSight

python scripts/rdp_pkl_to_zarr.py --pkl-dir /home/yfx/turbo/Dataset/wipe_zr/pkl -o /home/yfx/turbo/Dataset/wipe_zr/zarr/replay_buffer.zarr --pca-dir /home/yfx/turbo/reactive_diffusion_policy/data/PCA_Transform_GelSight
# 可选：--pca-dim 15（默认 15）
```

### zarr 结构

- `data/`：`timestamp`, `camera_f`, `camera_r`, `Tactile_Image_Right`, `Marker_Tracking_Right_DXDY`（126 维）, `joint_states_single`（展平为 position; velocity; effort）, `action`（下一帧 joint position）
- 若指定 `--pca-dir`：还会写入 `Marker_Tracking_Right_DXDY_emb`（15 维，用于训练时触觉观测）
- `meta/episode_ends`：每段 episode 的结束下标

---

## 3. 训练

### dataset_path

训练 config 里 `task.dataset_path` 指向 **zarr 所在目录**（即 `replay_buffer.zarr` 的父目录），例如：

- zarr 路径：`data/my_task/replay_buffer.zarr`
- 则 `dataset_path: data/my_task`

### shape_meta

`shape_meta` 要和 zarr 里的 key 一致。你当前 pkl/zarr 的 key 是：

- **obs**：`camera_f`, `camera_r`, `Tactile_Image_Right`（rgb）, `Marker_Tracking_Right_DXDY`（low_dim, 126）, 若用了 `--pca-dir` 还有 `Marker_Tracking_Right_DXDY_emb`（low_dim, 15）, `joint_states_single`（low_dim, 展平后维度）
- **action**：关节 position 维度（与 joint_ctrl_single position 长度一致）

示例（需按你实际图像尺寸和关节数改）。**用 PCA 时**建议用 `Marker_Tracking_Right_DXDY_emb` 作为触觉观测：

```yaml
shape_meta:
  obs:
    camera_f:
      shape: [3, H, W]   # 你的 camera_f 分辨率
      type: rgb
    camera_r:
      shape: [3, H, W]
      type: rgb
    Tactile_Image_Right:
      shape: [3, H_t, W_t]
      type: rgb
    # 用 PCA 时用 _emb（15 维），不用 PCA 时用下面 126 维
    Marker_Tracking_Right_DXDY_emb:
      shape: [15]
      type: low_dim
    # Marker_Tracking_Right_DXDY:  # 不用 PCA 时用这个
    #   shape: [126]
    #   type: low_dim
    joint_states_single:
      shape: [N]   # N = position_dim + velocity_dim + effort_dim
      type: low_dim
  action:
    shape: [J]    # J = 关节 position 维度
```

### 训练命令

```bash
python train.py --config-name=train_diffusion_unet_real_image_workspace task=你的任务名
```

任务名对应 `reactive_diffusion_policy/config/task/` 下你新建或复制的 yaml（其中 `dataset_path` 和 `shape_meta` 如上）。

---

## 触觉 PCA

在 pkl→zarr 时用 `--pca-dir` 即可做 126→15 降维，写入 `Marker_Tracking_Right_DXDY_emb`。PCA 矩阵需与 RDP 一致（如 `data/PCA_Transform_McTAC_v1` 下的 `pca_transform_matrix.npy`、`pca_mean_matrix.npy`）。训练时在 shape_meta 里用 `Marker_Tracking_Right_DXDY_emb`（15 维）作为触觉观测即可。
