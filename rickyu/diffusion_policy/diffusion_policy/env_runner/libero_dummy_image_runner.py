"""No-op image runner for LIBERO + Diffusion Policy training when robosuite rollout is not set up."""

from typing import Dict

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class LiberoDummyImageRunner(BaseImageRunner):
    """
    Returns empty logs so TrainDiffusionUnetImageWorkspace can run without RobomimicImageRunner.

    Use this while training on converted LIBERO HDF5; switch to a LIBERO/robosuite rollout
    runner once the environment is installed and env_args paths are valid.
    """

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
