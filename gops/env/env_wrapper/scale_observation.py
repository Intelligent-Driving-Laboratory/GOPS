from __future__ import annotations

from typing import TypeVar, Tuple, Union

import gym
import numpy as np
import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.env.env_wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ScaleObservationData(gym.Wrapper):
    """
        obs_rescaled = (obs + shift) * scale
        info["raw_obs"] = obs`
        example: add following to example script
            parser.add_argument("--obs_scale", default=np.array([2, 2, 2, 2]))
            parser.add_argument("--obs_shift", default=np.array([0, 0, 0, 0]))
    """
    def __init__(self, env, shift: Union[np.ndarray, float, list] = 0.0, scale: Union[np.ndarray, float, list] = 1.0):
        super(ScaleObservationData, self).__init__(env)
        if isinstance(shift, list):
            shift = np.array(shift, dtype=np.float32)
        if isinstance(scale, list):
            scale = np.array(scale, dtype=np.float32)
        self.shift = shift
        self.scale = scale

    def observation(self, observation):
        return (observation + self.shift) * self.scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_scaled = self.observation(obs)
        info["raw_obs"] = obs
        return obs_scaled, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        obs_scaled = self.observation(obs)
        info["raw_obs"] = obs
        return obs_scaled, r, d, info


class ScaleObservationModel(ModelWrapper):
    """
        obs_rescaled = (obs + shift) * scale
        example: add following to example script
            parser.add_argument("--obs_scale", default=np.array([2, 2, 2, 2]))
            parser.add_argument("--obs_shift", default=np.array([0, 0, 0, 0]))
    """
    def __init__(self,
                 model: PythBaseModel,
                 shift: Union[np.ndarray, float, list] = 0.0,
                 scale: Union[np.ndarray, float, list] = 1.0
                 ):
        super(ScaleObservationModel, self).__init__(model)
        if isinstance(shift, np.ndarray) or isinstance(shift, list):
            shift = torch.as_tensor(shift, dtype=torch.float32, device=self.model.device)
        if isinstance(scale, np.ndarray) or isinstance(scale, list):
            scale = torch.as_tensor(scale, dtype=torch.float32, device=self.model.device)
        self.shift = shift
        self.scale = scale

    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        unscaled_obs = obs / self.scale - self.shift
        next_obs, reward, next_done, next_info = self.model.forward(unscaled_obs, action, done, info)
        scaled_next_obs = (next_obs + self.shift) * self.scale
        return scaled_next_obs, reward, next_done, next_info
