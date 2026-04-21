#!/bin/bash

# DP w. GelSight Emb. (Peeling)
#python eval_real_robot_flexiv.py \
#      --config-name train_diffusion_unet_real_image_workspace \
#      task=real_peel_image_gelsight_emb_absolute_12fps \
#      +task.env_runner.output_dir=/path/for/saving/videos \
#      +ckpt_path=/path/to/dp/checkpoint




# RDP w. Force (Peeling)
# python eval_real_robot_flexiv.py \
#       --config-name train_latent_diffusion_unet_real_image_workspace \
#       task=real_peel_image_wrench_ldp_24fps \
#       at=at_peel \
#       at_load_dir=/path/to/at/checkpoint \
#       +task.env_runner.output_dir=/path/for/saving/videos \
#       +ckpt_path=/path/to/ldp/checkpoint

export HYDRA_FULL_ERROR=1

python eval_piper.py \
    --config-name train_latent_diffusion_unet_real_image_workspace \
    task=real_liftbottle_image_gelsight_emb_ldp_24fps \
    at=at_liftbottle \
    at_load_dir="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.19/12.44.26_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0219124423/checkpoints/latest.ckpt" \
    +ckpt_path="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.19/14.05.32_train_latent_diffusion_unet_image_real_liftbottle_image_gelsight_emb_ldp_24fps_0219124423/checkpoints/latest.ckpt"\

###80 eposide
    # at_load_dir="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.19/12.44.26_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0219124423/checkpoints/latest.ckpt" \
    # +ckpt_path="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.19/14.05.32_train_latent_diffusion_unet_image_real_liftbottle_image_gelsight_emb_ldp_24fps_0219124423/checkpoints/latest.ckpt"\

###50 eposide
    # at_load_dir="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.21/14.08.46_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0221140839/checkpoints/latest.ckpt" \
    # +ckpt_path="/home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.21/15.01.31_train_latent_diffusion_unet_image_real_liftbottle_image_gelsight_emb_ldp_24fps_0221140839/checkpoints/latest.ckpt"\

###80 eposide 600 epochs
# /home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.22/16.53.12_train_latent_diffusion_unet_image_real_liftbottle_image_gelsight_emb_ldp_24fps_0222153041/checkpoints/latest.ckpt
# /home/yfx/turbo/reactive_diffusion_policy/data/outputs/2026.02.21/14.08.46_train_vae_real_liftbottle_image_gelsight_emb_at_24fps_0221140839/checkpoints/epoch=0490-train_loss=0.008765.ckpt