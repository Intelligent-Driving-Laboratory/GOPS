from __future__ import annotations

from typing import TypeVar, Tuple, Union
import numpy as np
import gym
import torch
import torch.nn as nn

from gops.env.env_wrapper.base import ModelWrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ScaleObservationData(gym.Wrapper):
    """
        obs_rescaled = (obs + shift) * scale
        info["raw_obs"] = obs
    """
    def __init__(self, env, shift: Union[np.ndarray, float] = 0.0, scale: Union[np.ndarray, float] = 1.0):
        super(ScaleObservationData, self).__init__(env)
        self.shift = shift
        self.scale = scale

    def observation(self, observation):
        return (observation + self.shift) * self.scale

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            obs_scaled = self.observation(obs)
            info["raw_obs"] = obs
            return obs_scaled, info
        else:
            return self.observation(self.env.reset(**kwargs))

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        obs_scaled = self.observation(obs)
        info["raw_obs"] = obs
        return obs_scaled, r, d, info


class ScaleObservationModel(ModelWrapper):
    """
        obs_rescaled = (obs + shift) * scale
    """
    def __init__(self,
                 model: nn.Module,
                 shift: Union[torch.Tensor, float] = 0.0,
                 scale: Union[torch.Tensor, float] = 1.0
                 ):
        super(ScaleObservationModel, self).__init__(model)
        self.shift = shift
        self.scale = scale

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done=None):
        unscaled_state = state / self.scale - self.shift
        s, r, d, info = self.model.forward(unscaled_state, action, beyond_done)
        scaled_state = (s + self.shift) * self.scale
        return scaled_state, r, d, info
