#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data and model type environment wrappers that scale observation
#  Update: 2022-09-21, Yuhang Zhang: create scale reward wrapper
#  Update: 2022-10-27, Yujie Yang: rewrite scale reward wrapper


from __future__ import annotations

from typing import Tuple, Union

import gym
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict
from gym.core import ObsType, ActType


class ShapingRewardData(gym.Wrapper):
    """RewardShaping wrapper for data type environments.

    :param env: data type environment.
    :param float reward_shift: shift factor.
    :param float reward_scale: scale factor.

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
    """RewardShaping wrapper for model type environments.

    :param PythBaseModel model: model type environment.
    :param float reward_shift: shift factor.
    :param float reward_scale: scale factor.

    r_rescaled = (r + shift) * scale
    info["raw_reward"] = r
    example: add following to example script
        parser.add_argument("--reward_scale", default=0.5)
        parser.add_argument("--reward_shift", default=0)
    """

    def __init__(
        self,
        model: PythBaseModel,
        reward_shift: Union[torch.Tensor, float] = 0.0,
        reward_scale: Union[torch.Tensor, float] = 1.0,
    ):
        super(ShapingRewardModel, self).__init__(model)
        self.shift = reward_shift
        self.scale = reward_scale

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs, reward, next_done, next_info = self.model.forward(
            obs, action, done, info
        )
        reward_scaled = (reward + self.shift) * self.scale
        return next_obs, reward_scaled, next_done, next_info
