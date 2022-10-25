import warnings
from typing import Tuple, Union

import torch
from torch import nn


class PythBaseModel(nn.Module):
    def __init__(self,
                 obs_lower_bound,
                 obs_upper_bound,
                 action_lower_bound,
                 action_upper_bound,
                 clamp_obs: bool = True,
                 clamp_action: bool = True,
                 done_mask: bool = True,
                 ):
        super(PythBaseModel, self).__init__()
        self.obs_lower_bound = torch.tensor(obs_lower_bound, dtype=torch.float32)
        self.obs_upper_bound = torch.tensor(obs_upper_bound, dtype=torch.float32)
        self.action_lower_bound = torch.tensor(action_lower_bound, dtype=torch.float32)
        self.action_upper_bound = torch.tensor(action_upper_bound, dtype=torch.float32)
        self.clamp_obs = clamp_obs
        self.clamp_action = clamp_action
        self.done_mask = done_mask

    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: dict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clamp_obs:
            obs_clamp = obs.clamp(self.obs_lower_bound, self.obs_upper_bound)
            if obs_clamp != obs:
                warnings.warn("Observation out of space!")
            obs = obs_clamp

        if self.clamp_action:
            action_clamp = action.clamp(self.action_lower_bound, self.action_upper_bound)
            if action_clamp != action:
                warnings.warn("Action out of space!")
            action = action_clamp

        next_obs, reward, next_done, next_info = self.step(obs, action, info)

        if self.done_mask:
            next_obs = ~done * next_obs + done * obs
            reward = ~done * reward

        return next_obs, reward, next_done, next_info

    def step(self, obs: torch.Tensor, action: torch.Tensor, info: dict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError

    def get_terminal_cost(self, obs: torch.Tensor) -> Union[torch.Tensor, None]:
        return None

    def get_constraint(self, obs: torch.Tensor) -> Union[torch.Tensor, None]:
        return None
