from __future__ import annotations
"""
Websocket server for remote latent diffusion inference.

This script is meant to run on HPC. It only returns latent actions:
`policy.predict_action(..., return_latent_action=True)`.
"""
import pathlib
import pickle
import time
import traceback
from typing import Dict
import dill
import hydra
import torch
import websockets.sync.server
from loguru import logger
from websockets.exceptions import ConnectionClosed

from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace


def _is_authorized(
    conn: websockets.sync.server.ServerConnection, api_key: str | None
) -> bool:
    if not api_key:
        return True
    try:
        auth_header = conn.request.headers.get("Authorization")
    except Exception:
        return False
    return auth_header == f"Api-Key {api_key}"


class LatentPolicyWebsocketServer:
    def __init__(
        self,
        policy: BaseImagePolicy,
        host: str,
        port: int,
        dataset_obs_temporal_downsample_ratio: int,
        api_key: str | None = None,
        log_every_n: int = 20,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._api_key = api_key
        self._dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self._log_every_n = max(1, int(log_every_n))
        self._infer_count = 0
        self._sum_model_infer_ms = 0.0
        self._sum_total_ms = 0.0
        self._metadata = {
            "mode": "latent_action_only",
            "dataset_obs_temporal_downsample_ratio": dataset_obs_temporal_downsample_ratio,
        }

    def serve_forever(self) -> None:
        logger.info(f"Starting websocket server at ws://{self._host}:{self._port}")
        with websockets.sync.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            server.serve_forever()

    def _handler(self, conn: websockets.sync.server.ServerConnection) -> None:
        if not _is_authorized(conn, self._api_key):
            conn.send(pickle.dumps({"error": "Unauthorized"}, protocol=pickle.HIGHEST_PROTOCOL))
            conn.close(code=1008, reason="Unauthorized")
            return

        conn.send(pickle.dumps(self._metadata, protocol=pickle.HIGHEST_PROTOCOL))
        logger.info(f"Client connected: {conn.remote_address}")

        while True:
            try:
                payload = conn.recv()
                if isinstance(payload, str):
                    conn.send(pickle.dumps({"error": "Expected binary payload"}))
                    continue

                request = pickle.loads(payload)
                cmd = request.get("cmd", "infer")
                if cmd == "metadata":
                    conn.send(pickle.dumps(self._metadata, protocol=pickle.HIGHEST_PROTOCOL))
                    continue
                if cmd == "reset":
                    self._policy.reset()
                    conn.send(pickle.dumps({"ok": True}, protocol=pickle.HIGHEST_PROTOCOL))
                    continue
                if cmd != "infer":
                    conn.send(pickle.dumps({"error": f"Unknown cmd: {cmd}"}))
                    continue

                obs_dict = request["obs"]
                ratio = int(
                    request.get(
                        "dataset_obs_temporal_downsample_ratio",
                        self._dataset_obs_temporal_downsample_ratio,
                    )
                )
                total_start = time.time()
                prep_start = time.time()
                torch_obs_dict = dict_apply(
                    obs_dict,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device=self._policy.device),
                )
                prep_ms = (time.time() - prep_start) * 1000.0
                model_start = time.time()
                with torch.no_grad():
                    action_dict = self._policy.predict_action(
                        torch_obs_dict,
                        dataset_obs_temporal_downsample_ratio=ratio,
                        return_latent_action=True,
                    )
                model_infer_ms = (time.time() - model_start) * 1000.0
                latent_action = action_dict["action"].detach().cpu().numpy().squeeze(0)
                total_ms = (time.time() - total_start) * 1000.0
                self._infer_count += 1
                self._sum_model_infer_ms += model_infer_ms
                self._sum_total_ms += total_ms
                if self._infer_count % self._log_every_n == 0:
                    avg_model_ms = self._sum_model_infer_ms / self._infer_count
                    avg_total_ms = self._sum_total_ms / self._infer_count
                    logger.info(
                        f"[LDP-HPC] count={self._infer_count} "
                        f"model_infer_ms={model_infer_ms:.2f} avg_model_ms={avg_model_ms:.2f} "
                        f"prep_ms={prep_ms:.2f} total_ms={total_ms:.2f} avg_total_ms={avg_total_ms:.2f}"
                    )
                response: Dict = {
                    "latent_action": latent_action,
                    "server_timing": {
                        "prep_ms": prep_ms,
                        "model_infer_ms": model_infer_ms,
                        "total_ms": total_ms,
                    },
                }
                conn.send(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
            except ConnectionClosed:
                logger.info(f"Client disconnected: {conn.remote_address}")
                break
            except Exception:
                err = traceback.format_exc()
                logger.error(err)
                try:
                    conn.send(pickle.dumps({"error": err}, protocol=pickle.HIGHEST_PROTOCOL))
                finally:
                    conn.close(code=1011, reason="Internal server error")
                break


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("reactive_diffusion_policy", "config")),
    config_name="train_latent_diffusion_unet_real_image_workspace",
)
def main(cfg) -> None:
    ckpt_path = cfg.ckpt_path
    if ckpt_path is None:
        raise ValueError("ckpt_path must be provided for serving latent diffusion policy.")

    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if "diffusion" not in cfg.name or "latent" not in cfg.name:
        raise NotImplementedError("This server supports latent diffusion policy only.")

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.at.set_normalizer(policy.normalizer)

    serve_device = cfg.get("serve_device", cfg.training.device)
    device = torch.device(serve_device)
    policy.eval().to(device)
    policy.num_inference_steps = int(cfg.get("serve_num_inference_steps", 8))

    env_runner_cfg = cfg.task.env_runner
    dataset_obs_temporal_downsample_ratio = int(
        cfg.get(
            "serve_dataset_obs_temporal_downsample_ratio",
            env_runner_cfg.dataset_obs_temporal_downsample_ratio,
        )
    )

    host = cfg.get("ws_host", "0.0.0.0")
    port = int(cfg.get("ws_port", 8765))
    api_key = cfg.get("ws_api_key", None)
    log_every_n = int(cfg.get("serve_log_every_n", 20))

    server = LatentPolicyWebsocketServer(
        policy=policy,
        host=host,
        port=port,
        dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio,
        api_key=api_key,
        log_every_n=log_every_n,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
