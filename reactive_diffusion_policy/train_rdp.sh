#!/bin/bash

GPU_ID=0

TASK_NAME="liftbottle"
DATASET_PATH="/nfs/turbo/coe-vkamat/Dataset/wipe_zr/zarr"
# DATASET_PATH="/home/yfx/turbo/Dataset/rdp_liftbottle"
LOGGING_MODE="offline"

TIMESTAMP=$(date +%m%d%H%M%S)
SEARCH_PATH="./data/outputs"

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training Asymmetric Tokenizer..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_at_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_at_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=real_${TASK_NAME}_image_gelsight_emb_at_24fps_${TIMESTAMP} \
    at=at_liftbottle \
    logging.mode=${LOGGING_MODE}
# training at/ task / at 


# # find the latest checkpoint
# echo ""
# echo "Searching for the latest AT checkpoint..."
AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${TIMESTAMP}*" -type d)/checkpoints/latest.ckpt
# AT_LOAD_DIR="/nfs/turbo/coe-vkamat/reactive_diffusion_policy/data/outputs/2026.02.10/17.23.11_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0210172310/checkpoints/latest.ckpt"
# if [ ! -f "${AT_LOAD_DIR}" ]; then
#     echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
#     exit 1
# fi
# AT_LOAD_DIR="/nfs/turbo/coe-vkamat/reactive_diffusion_policy/data/outputs/2026.02.05/14.39.47_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0205143945/checkpoints/latest.ckpt"
# Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=real_${TASK_NAME}_image_gelsight_emb_ldp_24fps_${TIMESTAMP} \
    at=at_liftbottle \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE}

###### at / ldp / task
###  42682088 