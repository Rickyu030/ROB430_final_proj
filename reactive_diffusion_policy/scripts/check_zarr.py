import zarr
import sys
#python scripts/check_zarr.py /home/yfx/turbo/Dataset/wipe_zr1/zarr/replay_buffer.zarr
# 替换为你的 .zarr 文件夹路径
zarr_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.zarr"

print(f"--- Inspecting: {zarr_path} ---")

try:
    # mode='r' 只读模式打开
    root = zarr.open_group(zarr_path, mode='r')
    
    # 打印树状结构 (包含 Key, Shape, Dtype, Chunk大小, 压缩方式)
    print(root.tree())

except Exception as e:
    print(f"Error: {e}")


# /
#  ├── data
#  │   ├── Marker_Tracking_Right_DXDY (19015, 126) float32
#  │   ├── Marker_Tracking_Right_DXDY_emb (19015, 15) float32
#  │   ├── Tactile_Image_Right (19015, 480, 640, 3) uint8
#  │   ├── action (19015, 7) float32
#  │   ├── camera_f (19015, 720, 1280, 3) uint8
#  │   ├── camera_r (19015, 720, 1280, 3) uint8
#  │   ├── joint_states_single (19015, 7) float32
#  │   └── timestamp (19015,) float32
#  └── meta
#      └── episode_ends (50,) int64