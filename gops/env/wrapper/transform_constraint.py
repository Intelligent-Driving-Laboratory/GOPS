#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data and model type environment wrappers that transfer constraint to punishment
#  Update: 2022-09-21, Yuhang Zhang: create constraint transfer wrapper


from typing import Tuple

import gym
import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.wrapper.base import ModelWrapper
from gym.core import ActType, ObsType
from gops.utils.gops_typing import InfoDict


class EnvC2U(gym.Wrapper):
    """Data type environment wrapper. Transform env with constraints to env without
        constraint by punishing constraint function in reward function.
    :param env: data type environment.
    :param float punish_factor: punish = punish_factor * constraint
    """

    def __init__(self, env, punish_factor: float = 10.0):
        super().__init__(env)
        self.punish_factor = punish_factor

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        punish = np.sum(self.punish_factor * np.clip(info["constraint"], 0, np.inf))
        reward_new = reward - punish
        return observation, reward_new, done, info


class ModelC2U(ModelWrapper):
    """Model type environment wrapper. Transform env with constraints to env without
        constraint by punishing constraint function in reward function.
    :param PythBaseModel model: model type environment.
    :param float punish_factor: punish = punish_factor * constraint
    """

    def __init__(self, model: PythBaseModel, punish_factor: float = 10.0):
        super(ModelC2U, self).__init__(model)
        self.model = model
        self.punish_factor = punish_factor

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        obs_next, reward, done, info = self.model.forward(obs, action, done, info)
        const = info["constraint"].reshape(obs_next.shape[0], -1)

        punish = torch.clamp(const, min=0) * self.punish_factor

        reward_new = reward - torch.sum(punish, dim=1)

        return obs_next, reward_new, done, info
