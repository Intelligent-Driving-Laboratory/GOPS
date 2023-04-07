#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: action repeat wrappers for data and model type environment
#  Update: 2022-11-15, Wenxuan Wang: create action repeat wrapper


from __future__ import annotations

from typing import TypeVar, Tuple, Union

import gym
import torch
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ActionRepeatData(gym.Wrapper):
    """Action repeat wrapper fot data type environments, repeat 'repeat_num' times
        action in one step and return last step observation.

    :param env: data type environment.
    :param int repeat_num: repeat n times action in one step.
    :param bool sum_reward: sum the rewards during repeating steps, if set to False,
        only use reward in last step.
    """

    def __init__(self, env, repeat_num: int = 1, sum_reward: bool = True):
        super(ActionRepeatData, self).__init__(env)
        self.repeat_num = repeat_num
        self.sum_reward = sum_reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        sum_r = 0
        for _ in range(self.repeat_num):
            obs, r, d, info = self.env.step(action)
            sum_r += r
            if d:
                break
        if not self.sum_reward:
            sum_r = r
        return obs, sum_r, d, info


class ActionRepeatModel(ModelWrapper):
    """Action repeat wrapper fot model type environments, repeat 'repeat_num' times
        action in one step and return last step observation.

    :param PythBaseModel model: gops model type environment.
    :param int repeat_num: repeat n times action in one step.
    :param bool sum_reward: sum the rewards during repeating steps, if set to False,
        only use reward in last step.
    """

    def __init__(
        self, model: PythBaseModel, repeat_num: int = 1, sum_reward: bool = True
    ):
        super(ActionRepeatModel, self).__init__(model)
        self.repeat_num = repeat_num
        self.sum_reward = sum_reward

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        sum_reward = 0
        for _ in range(self.repeat_num):
            next_obs, reward, next_done, next_info = self.model.forward(
                obs, action, done, info
            )
            sum_reward += reward
            obs, done, info = next_obs, done, info
        if not self.sum_reward:
            sum_reward = reward
        return next_obs, sum_reward, next_done, next_info
