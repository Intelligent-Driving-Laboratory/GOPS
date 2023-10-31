#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: base wrapper for model type environments
#  Update: 2022-09-21, Yuhang Zhang: create base wrapper
#  Update: 2022-10-26, Yujie Yang: rewrite base wrapper


from typing import Tuple

import gym
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class ModelWrapper:
    """Base wrapper class for model type environment wrapper.

    :param PythBaseModel model: gops model type environment.
    """

    def __init__(self, model: PythBaseModel):
        self.model = model
        self.forward_flag = False

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        return self.model.forward(obs, action, done, info)
    
    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model.get_next_state(state, action)
    
    def robot_model_get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model.robot_model.get_next_state(robot_state, action)
    
    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.model.get_reward(state, action)

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def unwrapped(self):
        return self.model.unwrapped


class ActionWrapper(gym.ActionWrapper):
    def step(self, action):
        raw_action = self.action(action)
        next_obs, reward, done, info = self.env.step(raw_action)
        info["raw_action"] = raw_action
        return next_obs, reward, done, info


class ActionModelWrapper(ModelWrapper):
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        self.forward_flag = True
        action = self.action(action)
        ret = super().forward(obs, action, done, info)
        self.forward_flag = False
        return ret
    
    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.forward_flag:
            action = self.action(action)
        return super().get_next_state(state, action)
    
    def robot_model_get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.forward_flag:
            action = self.action(action)
        return super().robot_model_get_next_state(robot_state, action)
    
    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.forward_flag:
            action = self.action(action)
        return super().get_reward(state, action)

    def action(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
