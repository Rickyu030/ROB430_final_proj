Inference

obs = get_obs_dict(rosnode)
feature1 = obs_encoder(obs)
cond_sample = np.cat(feature1,feature2,...) 
pred_noise = latent_dp(cond_sample)

for i in range(inference_step):
    latent_variable = gassuian - pred_noise

latent_action =  latent_variable
feature_at = np.cat(gelsight_feature,latent_action,future_action_chunk) 
#future_action_chunk.shape=[dim,16]
#gelsight_fearture = [dim,15]
#latent_action = [dim,7*8]
action_chunk = gru_decoder(fearture_at)  #action_chunk = [32,7]
#action_chunk[:15] = current_action_chunk  action_chunk[16:] = future_action_chunk

4Hz
16Hz
LDP_step = 4
AT_step = 16
interval_step = 16
def action_thread:
    latent_variable = get_obs(ensemble_buffer)
    actionchunk = at(fearture_at)

def main:
    step = 0
    while true:
        latent_variable = ldp(feature_ldp)
        if step // interval_step == 0:
            ensemble_buffer.add(latent_variable)

        step += AT_step/LDP_step
