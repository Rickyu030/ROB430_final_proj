#!/usr/bin/env python3
"""
将 ROS1 rosbag 转为 pkl，按原始 topic 名作为 key 存储（不改成 RDP 的 leftWristCameraRGB 等）。

存储 key 与 topic 对应关系（可改下面常量）:
  camera_f          <- /camera_f/color/image_raw
  camera_r          <- /camera_r/color/image_raw
  Tactile_Image_Right <- /Tactile_Image_Right
  Marker_Tracking_Right_DXDY <- /Marker_Tracking_Right_DXDY (126 维，存成 (63,2) 或 (126,))
  joint_states_single <- /right_arm/joint_states_single (position/velocity/effort)
  joint_ctrl_single  <- /right_arm/joint_ctrl_single (同上)

依赖: ROS1 (rosbag), cv_bridge, numpy。
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

# ROS1
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

# ----------------- 可配置常量：topic 与 pkl 中 key 一一对应，不改成别的名字 -----------------
TOPIC_CAMERA_F = "/camera_f/color/image_raw"
TOPIC_CAMERA_R = "/camera_r/color/image_raw"
TOPIC_TACTILE_IMAGE = "/Tactile_Image_Right"
TOPIC_MARKER_DXDY = "/Marker_Tracking_Right_DXDY"
TOPIC_JOINT_STATES = "/right_arm/joint_states_single"
TOPIC_JOINT_CTRL = "/right_arm/joint_ctrl_single"

# pkl 里用的 key，与上面 topic 对应：camera_r 就存成 camera_r
KEY_CAMERA_F = "camera_f"
KEY_CAMERA_R = "camera_r"
KEY_TACTILE_IMAGE = "Tactile_Image_Right"
KEY_MARKER_DXDY = "Marker_Tracking_Right_DXDY"
KEY_JOINT_STATES = "joint_states_single"
KEY_JOINT_CTRL = "joint_ctrl_single"

MARKER_DIM = 126  # 63*2
MARKER_SHAPE = (63, 2)

# 以哪条 topic 的时间戳为「主时钟」做对齐
ALIGN_REF_TOPIC = TOPIC_MARKER_DXDY


def _get_stamp(msg):
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return msg.header.stamp.to_sec()
    return 0.0


def _nearest_msg(messages, t):
    """从 (stamp, msg) 列表中取时间最接近 t 的一条。"""
    if not messages:
        return None
    idx = np.searchsorted([s for s, _ in messages], t, side="left")
    if idx == 0:
        return messages[0][1]
    if idx >= len(messages):
        return messages[-1][1]
    return messages[idx - 1][1] if t - messages[idx - 1][0] <= messages[idx][0] - t else messages[idx][1]


def _joint_state_to_array(msg):
    """JointState 只取 position，存成一条一维数组。用于 joint_states_single 和 joint_ctrl_single。"""
    if msg is None:
        return np.array([], dtype=np.float64)
    arr = np.array(list(getattr(msg, "position", [])), dtype=np.float64)
    return arr


def load_topic_messages(bag_path: str, topic: str, msg_type):
    """从 bag 里读取指定 topic 的 (stamp, msg) 列表。"""
    out = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            stamp = t.to_sec() if hasattr(t, "to_sec") else _get_stamp(msg)
            out.append((stamp, msg))
    out.sort(key=lambda x: x[0])
    return out


def _img_to_array(msg, bridge: CvBridge):
    """从 Image 转成 numpy，分辨率与 rosbag 中一致，不做 resize。"""
    if msg is None:
        return None
    try:
        arr = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr
    except Exception:
        return None


def _get_first_valid_image_shape(msgs: list, bridge: CvBridge) -> tuple | None:
    """从该 topic 的消息列表中取第一个能成功解码的图像，返回 (H, W, C)，用于缺失时的 fallback 尺寸。"""
    for _, msg in msgs:
        arr = _img_to_array(msg, bridge)
        if arr is not None and arr.size > 0:
            return arr.shape
    return None


def build_frame_dict(
    t_ref: float,
    cam_f_msgs: list,
    cam_r_msgs: list,
    tactile_msgs: list,
    marker_msgs: list,
    joint_states_msgs: list,
    joint_ctrl_msgs: list,
    bridge: CvBridge,
    fallback_shapes: dict,
) -> dict | None:
    """
    按参考时间 t_ref 对齐各 topic，拼成一帧的 dict。
    图像分辨率保持 rosbag 原始尺寸；仅当某帧缺失或解码失败时用 fallback_shapes 对应尺寸的零图填充。
    """
    marker_msg = _nearest_msg(marker_msgs, t_ref)
    if marker_msg is None:
        return None

    img_f = _img_to_array(_nearest_msg(cam_f_msgs, t_ref), bridge)
    img_r = _img_to_array(_nearest_msg(cam_r_msgs, t_ref), bridge)
    img_tactile = _img_to_array(_nearest_msg(tactile_msgs, t_ref), bridge)

    def _fallback_img(key, img):
        if img is not None and img.size > 0:
            return img
        shape = fallback_shapes.get(key)
        if shape is not None:
            return np.zeros(shape, dtype=np.uint8)
        return np.zeros((1, 1, 3), dtype=np.uint8)  # 该 topic 从未成功解码时的最小占位

    marker_data = np.array(marker_msg.data, dtype=np.float32)
    if marker_data.size != MARKER_DIM:
        marker_data = np.resize(marker_data, MARKER_DIM)
    marker_arr = marker_data.reshape(MARKER_SHAPE).copy()

    frame = {
        "timestamp": t_ref,
        KEY_CAMERA_F: _fallback_img(KEY_CAMERA_F, img_f),
        KEY_CAMERA_R: _fallback_img(KEY_CAMERA_R, img_r),
        KEY_TACTILE_IMAGE: _fallback_img(KEY_TACTILE_IMAGE, img_tactile),
        KEY_MARKER_DXDY: marker_arr,
        KEY_JOINT_STATES: _joint_state_to_array(_nearest_msg(joint_states_msgs, t_ref)),
        KEY_JOINT_CTRL: _joint_state_to_array(_nearest_msg(joint_ctrl_msgs, t_ref)),
    }
    return frame


def run(bag_path: str, out_pkl_path: str, ref_topic: str | None = None) -> None:
    ref_topic = ref_topic or ALIGN_REF_TOPIC
    bridge = CvBridge()

    # 读取各 topic
    marker_msgs = load_topic_messages(bag_path, TOPIC_MARKER_DXDY, Float32MultiArray)
    if not marker_msgs:
        raise RuntimeError(f"Bag 中未找到 {TOPIC_MARKER_DXDY}，请检查 topic 名与消息类型。")

    cam_f_msgs = load_topic_messages(bag_path, TOPIC_CAMERA_F, Image)
    cam_r_msgs = load_topic_messages(bag_path, TOPIC_CAMERA_R, Image)
    tactile_msgs = load_topic_messages(bag_path, TOPIC_TACTILE_IMAGE, Image)
    joint_states_msgs = load_topic_messages(bag_path, TOPIC_JOINT_STATES, JointState)
    joint_ctrl_msgs = load_topic_messages(bag_path, TOPIC_JOINT_CTRL, JointState)

    # 用本 bag 中各 topic 首次成功解码的图像尺寸作为「缺失时」的 fallback，不写死 480×640 等
    fallback_shapes = {}
    for key, msgs in [
        (KEY_CAMERA_F, cam_f_msgs),
        (KEY_CAMERA_R, cam_r_msgs),
        (KEY_TACTILE_IMAGE, tactile_msgs),
    ]:
        sh = _get_first_valid_image_shape(msgs, bridge)
        if sh is not None:
            fallback_shapes[key] = sh

    ref_timestamps = [t for t, _ in marker_msgs]
    frames = []
    for t_ref in ref_timestamps:
        frame = build_frame_dict(
            t_ref,
            cam_f_msgs,
            cam_r_msgs,
            tactile_msgs,
            marker_msgs,
            joint_states_msgs,
            joint_ctrl_msgs,
            bridge,
            fallback_shapes,
        )
        if frame is not None:
            frames.append(frame)

    os.makedirs(os.path.dirname(os.path.abspath(out_pkl_path)) or ".", exist_ok=True)

    out_data = SimpleNamespace(frames=frames)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(out_data, f)

    print(f"已写入 {len(frames)} 帧 -> {out_pkl_path}（key: {KEY_CAMERA_F}, {KEY_CAMERA_R}, {KEY_TACTILE_IMAGE}, {KEY_MARKER_DXDY}, {KEY_JOINT_STATES}, {KEY_JOINT_CTRL}）")


def main():
    parser = argparse.ArgumentParser(
        description="ROS1 rosbag -> pkl，按原始 topic 名存成 key（camera_r 即 camera_r）。支持单 bag 或目录下多个 bag。"
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("bag_path", type=str, nargs="?", help="单个 rosbag 路径")
    g.add_argument(
        "--bag-dir",
        type=str,
        metavar="DIR",
        help="包含多个 .bag 的目录，按文件名排序后逐个转成 pkl（每个 bag 一个 pkl）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出：单 bag 时为 pkl 路径；--bag-dir 时为输出目录（默认与 bag-dir 同目录）",
    )
    parser.add_argument(
        "--ref-topic",
        type=str,
        default=None,
        help=f"对齐用的主时钟 topic，默认 {ALIGN_REF_TOPIC}",
    )
    args = parser.parse_args()
    ref_topic = args.ref_topic
    if args.bag_dir:
        bag_dir = os.path.expanduser(args.bag_dir)
        bag_paths = sorted(
            os.path.join(bag_dir, f)
            for f in os.listdir(bag_dir)
            if f.endswith(".bag")
        )
        if not bag_paths:
            raise SystemExit(f"未在目录下找到 .bag 文件: {bag_dir}")
        out_dir = os.path.expanduser(args.output) if args.output else bag_dir
        os.makedirs(out_dir, exist_ok=True)
        for bag_path in bag_paths:
            base = os.path.splitext(os.path.basename(bag_path))[0]
            out = os.path.join(out_dir, base + ".pkl")
            run(bag_path, out, ref_topic=ref_topic)
    else:
        out = args.output
        if not out:
            out = os.path.splitext(args.bag_path)[0] + ".pkl"
        else:
            out = os.path.expanduser(out)
        run(args.bag_path, out, ref_topic=ref_topic)


if __name__ == "__main__":
    main()
