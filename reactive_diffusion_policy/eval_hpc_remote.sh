#!/bin/bash
set -euo pipefail

export HYDRA_FULL_ERROR=1

################################################################################
# 1) Run this on HPC (latent diffusion inference server)
################################################################################
# python serve_latent_policy_ws.py \
#   --config-name train_latent_diffusion_unet_real_image_workspace \
#   task=real_liftbottle_image_gelsight_emb_ldp_24fps \
#   at=at_liftbottle \
#   at_load_dir="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.21/14.08.46_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0221140839/checkpoints/latest.ckpt" \
#   +ckpt_path="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.21/15.01.31_train_latent_diffusion_unet_image_real_liftbottle_image_gelsight_emb_ldp_24fps_0221140839/checkpoints/latest.ckpt" \
#   +ws_host=0.0.0.0 \
#   +ws_port=8765 \
#   +serve_device=cuda:0 \
#   +serve_num_inference_steps=8 \
#   +serve_log_every_n=20

################################################################################
# 2) Run this locally (ROS + AT decode local, latent inference remote)
################################################################################
python eval_piper_hpc_remote.py \
  --config-name train_latent_diffusion_unet_real_image_workspace \
  task=real_liftbottle_image_gelsight_emb_ldp_24fps \
  at=at_liftbottle \
  at_load_dir="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.19/12.44.26_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0219124423/checkpoints/latest.ckpt" \
  +ckpt_path="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.19/14.05.32_train_latent_diffusion_unet_image_real_liftbottle_image_gelsight_emb_ldp_24fps_0219124423/checkpoints/latest.ckpt"\
  +remote_host=127.0.0.1 \
  +remote_port=18765 \
  +local_at_device=cuda:0 \
  +client_log_every_n=20


#ssh -N -o ExitOnForwardFailure=yes -L 18765:gl1514:8765 zrrui@greatlakes.arc-ts.umich.edu