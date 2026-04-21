python scripts/rosbag_to_rdp_pkl.py --bag-dir /nfs/turbo/coe-vkamat/Dataset/wipe_zr1 -o /nfs/turbo/coe-vkamat/Dataset/wipe_zr1/pkl
python scripts/rdp_pkl_to_zarr.py --pkl-dir /nfs/turbo/coe-vkamat/Dataset/wipe_zr1 -o /nfs/turbo/coe-vkamat/Dataset/wipe_zr1/zarr/replay_buffer.zarr --pca-dir /nfs/turbo/coe-vkamat/reactive_diffusion_policy/data/PCA_Transform_GelSight
