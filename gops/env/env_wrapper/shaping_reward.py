from __future__ import annotations

from typing import TypeVar, Tuple, Union

import gym
import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.env.env_wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ShapingRewardData(gym.Wrapper):
    """
        r_rescaled = (r + reward_shift) * reward_scale
        info["raw_reward"] = r
        example: add following to example script
            parser.add_argument("--reward_scale", default=0.5)
            parser.add_argument("--reward_shift", default=0)
    """
    def __init__(self, env, reward_shift: float = 0.0, reward_scale: float = 1.0):
        super(ShapingRewardData, self).__init__(env)
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        r_scaled = (r + self.reward_shift) * self.reward_scale
        info["raw_reward"] = r
        return obs, r_scaled, d, info


class ShapingRewardModel(ModelWrapper):
    """
        r_rescaled = (r + reward_shift) * reward_scale
        example: add following to example script
            parser.add_argument("--reward_scale", default=0.5)
            parser.add_argument("--reward_shift", default=0)
    """

    def __init__(self,
                 model: PythBaseModel,
                 shift: Union[torch.Tensor, float] = 0.0,
                 scale: Union[torch.Tensor, float] = 1.0
                 ):
        super(ShapingRewardModel, self).__init__(model)
        self.shift = shift
        self.scale = scale

    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs, reward, next_done, next_info = self.model.forward(obs, action, done, info)
        reward_scaled = (reward + self.shift) * self.scale
        return next_obs, reward_scaled, next_done, next_info
