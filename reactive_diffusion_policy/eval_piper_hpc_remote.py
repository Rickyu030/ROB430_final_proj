"""
Evaluate with remote latent diffusion inference on HPC and local AT decoding.

Flow:
1) Local machine subscribes observations (same as eval_piper.py).
2) Local sends processed obs to HPC websocket server.
3) HPC returns latent actions.
4) Local machine decodes latent actions with AT and publishes robot actions.
"""
import gc
import pathlib
import pickle
import signal
import time
from typing import Dict, Optional, Tuple

import dill
import hydra
import torch
import websockets.sync.client
from loguru import logger
from omegaconf import DictConfig

from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace

from eval_piper import CustomRealRunner


class WebsocketLatentClient:
    def __init__(
        self,
        host: str,
        port: int,
        api_key: Optional[str] = None,
        retry_interval_sec: float = 5.0,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._api_key = api_key
        self._retry_interval_sec = retry_interval_sec
        self._ws, self._metadata = self._wait_for_server()

    @property
    def metadata(self) -> Dict:
        return self._metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info(f"Waiting for latent server at {self._uri} ...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata_raw = conn.recv()
                if isinstance(metadata_raw, str):
                    raise RuntimeError(f"Server error during handshake:\n{metadata_raw}")
                metadata = pickle.loads(metadata_raw)
                if "error" in metadata:
                    raise RuntimeError(f"Server rejected connection: {metadata['error']}")
                logger.info(f"Connected to latent server: {metadata}")
                return conn, metadata
            except Exception as e:
                logger.warning(f"Latent server unavailable ({e}), retry in {self._retry_interval_sec}s")
                time.sleep(self._retry_interval_sec)

    def infer(self, obs: Dict) -> Dict:
        req = {"cmd": "infer", "obs": obs}
        self._ws.send(pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server returned text error:\n{response}")
        payload = pickle.loads(response)
        if "error" in payload:
            raise RuntimeError(f"Inference server error:\n{payload['error']}")
        return payload

    def reset(self) -> None:
        self._ws.send(pickle.dumps({"cmd": "reset"}, protocol=pickle.HIGHEST_PROTOCOL))
        _ = self._ws.recv()

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


class LocalATDecoder:
    def __init__(
        self,
        at,
        normalizer,
        action_dim: int,
        n_obs_steps: int,
        n_action_steps: int,
        original_horizon: int,
        device: torch.device,
    ) -> None:
        self.at = at
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.original_horizon = original_horizon
        self.device = device

        self.normalizer.to(self.device)
        self.at.to(self.device)
        self.at.set_normalizer(self.normalizer)
        self.at.eval()

    @torch.no_grad()
    def predict_from_latent_action(
        self,
        latent_action: torch.Tensor,
        extended_obs_dict: Dict[str, torch.Tensor],
        extended_obs_last_step: int,
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
    ) -> Dict[str, torch.Tensor]:
        latent_action = latent_action.to(self.device)
        extended_obs_dict = dict_apply(extended_obs_dict, lambda x: x.to(self.device))

        if self.at.use_rnn_decoder:
            if extend_obs_pad_after:
                extend_obs_pad_after_n = self.original_horizon - extended_obs_last_step
            else:
                extend_obs_pad_after_n = None
            temporal_cond = self.at.get_temporal_cond(
                extended_obs_dict,
                extended_obs_last_step,
                extend_obs_pad_after_n=extend_obs_pad_after_n,
            )
            temporal_cond = temporal_cond.to(self.device)
            nsample = self.at.get_action_from_latent_with_temporal_cond(latent_action, temporal_cond)
        else:
            nsample = self.at.get_action_from_latent(latent_action)

        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        to = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        start = to - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}


class RemoteLatentHybridPolicy:
    """
    Policy facade for CustomRealRunner:
    - predict_action: remote latent diffusion (HPC).
    - predict_from_latent_action: local AT decoding.
    """

    def __init__(
        self,
        latent_client: WebsocketLatentClient,
        local_decoder: LocalATDecoder,
        client_log_every_n: int = 20,
    ) -> None:
        self._latent_client = latent_client
        self._local_decoder = local_decoder
        self.at = local_decoder.at
        self._device = torch.device("cpu")
        self._client_log_every_n = max(1, int(client_log_every_n))
        self._infer_count = 0
        self._sum_model_infer_ms = 0.0
        self._sum_total_ms = 0.0

    @property
    def device(self) -> torch.device:
        # Keep obs tensors on CPU before sending over websocket.
        return self._device

    def eval(self):
        return self

    def reset(self) -> None:
        self._latent_client.reset()

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int,
        return_latent_action: bool = False,
    ) -> Dict[str, torch.Tensor]:
        np_obs_dict = dict_apply(obs_dict, lambda x: x.squeeze(0).detach().cpu().numpy())
        payload = self._latent_client.infer(np_obs_dict)
        latent_action = payload["latent_action"]
        server_timing = payload.get("server_timing", {})
        model_infer_ms = server_timing.get("model_infer_ms", None)
        total_ms = server_timing.get("total_ms", None)
        if model_infer_ms is not None and total_ms is not None:
            self._infer_count += 1
            self._sum_model_infer_ms += float(model_infer_ms)
            self._sum_total_ms += float(total_ms)
            if self._infer_count % self._client_log_every_n == 0:
                logger.info(
                    f"[LDP-HPC-FromLocal] count={self._infer_count} "
                    f"model_infer_ms={float(model_infer_ms):.2f} "
                    f"avg_model_ms={self._sum_model_infer_ms / self._infer_count:.2f} "
                    f"total_ms={float(total_ms):.2f} "
                    f"avg_total_ms={self._sum_total_ms / self._infer_count:.2f}"
                )
        latent_action_t = torch.from_numpy(latent_action).unsqueeze(0)
        if not return_latent_action:
            raise NotImplementedError("Remote latent policy only supports return_latent_action=True.")
        return {"action": latent_action_t, "action_pred": latent_action_t}

    @torch.no_grad()
    def predict_from_latent_action(
        self,
        latent_action: torch.Tensor,
        extended_obs_dict: Dict[str, torch.Tensor],
        extended_obs_last_step: int,
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self._local_decoder.predict_from_latent_action(
            latent_action=latent_action,
            extended_obs_dict=extended_obs_dict,
            extended_obs_last_step=extended_obs_last_step,
            dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio,
            extend_obs_pad_after=extend_obs_pad_after,
        )


_runner_instance: Optional[CustomRealRunner] = None


def signal_handler(signum, frame):
    global _runner_instance
    logger.warning("\nSIGINT (Ctrl+C) received! Initiating graceful shutdown...")
    if _runner_instance is not None:
        _runner_instance.shutdown_requested = True
        _runner_instance.stop_event.set()
    raise KeyboardInterrupt


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("reactive_diffusion_policy", "config")),
    config_name="train_latent_diffusion_unet_real_image_workspace",
)
def main(cfg):
    global _runner_instance
    signal.signal(signal.SIGINT, signal_handler)

    ckpt_path = cfg.ckpt_path
    if ckpt_path is None:
        raise ValueError("ckpt_path must be provided.")

    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if "diffusion" not in cfg.name or "latent" not in cfg.name:
        raise NotImplementedError("This script only supports latent diffusion configs.")

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.at.set_normalizer(policy.normalizer)
    policy.eval()

    local_at_device = torch.device(
        cfg.get("local_at_device", "cuda:0" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Local AT device: {local_at_device}")
    local_decoder = LocalATDecoder(
        at=policy.at,
        normalizer=policy.normalizer,
        action_dim=policy.action_dim,
        n_obs_steps=policy.n_obs_steps,
        n_action_steps=policy.n_action_steps,
        original_horizon=policy.original_horizon,
        device=local_at_device,
    )

    remote_host = cfg.get("remote_host", "127.0.0.1")
    remote_port = int(cfg.get("remote_port", 8765))
    remote_api_key = cfg.get("remote_api_key", None)
    client_log_every_n = int(cfg.get("client_log_every_n", 20))
    latent_client = WebsocketLatentClient(
        host=remote_host,
        port=remote_port,
        api_key=remote_api_key,
    )
    hybrid_policy = RemoteLatentHybridPolicy(
        latent_client=latent_client,
        local_decoder=local_decoder,
        client_log_every_n=client_log_every_n,
    )

    task_cfg = cfg.task
    env_runner_cfg = task_cfg.env_runner
    if "action_ensemble_buffer_params" in env_runner_cfg:
        action_ensemble_buffer_params = env_runner_cfg.action_ensemble_buffer_params
    elif "tcp_ensemble_buffer_params" in env_runner_cfg:
        action_ensemble_buffer_params = env_runner_cfg.tcp_ensemble_buffer_params
    else:
        action_ensemble_buffer_params = DictConfig({"ensemble_mode": "new"})

    if "latent_action_ensemble_buffer_params" in env_runner_cfg:
        latent_action_ensemble_buffer_params = env_runner_cfg.latent_action_ensemble_buffer_params
    elif "latent_tcp_ensemble_buffer_params" in env_runner_cfg:
        latent_action_ensemble_buffer_params = env_runner_cfg.latent_tcp_ensemble_buffer_params
    else:
        latent_action_ensemble_buffer_params = DictConfig({"ensemble_mode": "new"})

    action_update_interval = env_runner_cfg.get(
        "action_update_interval",
        env_runner_cfg.get("tcp_action_update_interval", 6),
    )
    action_clip_range = env_runner_cfg.get("action_clip_range", None)
    joint_names = env_runner_cfg.get("joint_names", None)

    topic_mapping = env_runner_cfg.get("topic_mapping", None)
    if topic_mapping is None:
        topic_mapping = {}
        for key in task_cfg.shape_meta.get("obs", {}).keys():
            if key == "camera_f":
                topic_mapping[key] = "/camera_f/color/image_raw"
            elif key == "camera_r":
                topic_mapping[key] = "/camera_r/color/image_raw"
            elif "joint" in key.lower() and "state" in key.lower():
                if "right_arm" not in key:
                    topic_mapping[key] = f"/right_arm/{key}"

    force_subscribe_topics = env_runner_cfg.get("force_subscribe_topics", None)
    if force_subscribe_topics is None:
        force_subscribe_topics = {"/Marker_Tracking_Right_DXDY": "Float32MultiArray"}
    elif "/Marker_Tracking_Right_DXDY" not in force_subscribe_topics:
        force_subscribe_topics["/Marker_Tracking_Right_DXDY"] = "Float32MultiArray"

    runner = CustomRealRunner(
        shape_meta=task_cfg.shape_meta,
        transform_params=task_cfg.transforms,
        topic_prefix="/",
        topic_mapping=topic_mapping,
        force_subscribe_topics=force_subscribe_topics,
        action_topic="/right_arm/joint_ctrl_single",
        joint_names=joint_names,
        action_ensemble_buffer_params=action_ensemble_buffer_params,
        latent_action_ensemble_buffer_params=latent_action_ensemble_buffer_params,
        use_latent_action_with_rnn_decoder=env_runner_cfg.use_latent_action_with_rnn_decoder,
        use_relative_action=task_cfg.dataset.relative_action,
        eval_episodes=env_runner_cfg.eval_episodes,
        max_duration_time=env_runner_cfg.max_duration_time,
        action_update_interval=action_update_interval,
        action_clip_range=action_clip_range,
        control_fps=env_runner_cfg.control_fps,
        inference_fps=env_runner_cfg.inference_fps,
        latency_step=env_runner_cfg.latency_step,
        n_obs_steps=env_runner_cfg.n_obs_steps,
        obs_temporal_downsample_ratio=env_runner_cfg.obs_temporal_downsample_ratio,
        dataset_obs_temporal_downsample_ratio=env_runner_cfg.dataset_obs_temporal_downsample_ratio,
        downsample_extended_obs=env_runner_cfg.downsample_extended_obs,
        task_name=task_cfg.name,
    )
    _runner_instance = runner

    try:
        runner.run(hybrid_policy)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt in main! Shutting down...")
        runner.shutdown_requested = True
        runner.stop_event.set()
    finally:
        logger.info("Evaluation finished. Performing final cleanup...")
        latent_client.close()

        if hasattr(runner, "_cleanup_memory"):
            runner._cleanup_memory()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        del runner
        del hybrid_policy
        del local_decoder
        if "workspace" in locals():
            del workspace
        if "policy" in locals():
            del policy

        for _ in range(5):
            collected = gc.collect()
            if collected == 0:
                break

        _runner_instance = None
        logger.info("Cleanup completed. Exiting...")


if __name__ == "__main__":
    main()
