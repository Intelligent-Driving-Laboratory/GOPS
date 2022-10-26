import warnings
from typing import Tuple, Union

import torch

from gops.utils.gops_typing import InfoDict


class PythBaseModel:
    def __init__(self,
                 obs_lower_bound,
                 obs_upper_bound,
                 action_lower_bound,
                 action_upper_bound,
                 clamp_obs: bool = True,
                 clamp_action: bool = True,
                 done_mask: bool = False,
                 device: Union[torch.device, str, None] = None,
                 ):
        super(PythBaseModel, self).__init__()
        self.obs_lower_bound = torch.tensor(obs_lower_bound, dtype=torch.float32, device=device)
        self.obs_upper_bound = torch.tensor(obs_upper_bound, dtype=torch.float32, device=device)
        self.action_lower_bound = torch.tensor(action_lower_bound, dtype=torch.float32, device=device)
        self.action_upper_bound = torch.tensor(action_upper_bound, dtype=torch.float32, device=device)
        self.clamp_obs = clamp_obs
        self.clamp_action = clamp_action
        self.done_mask = done_mask
        self.device = device

    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        if self.clamp_obs:
            obs_clamp = obs.clamp(self.obs_lower_bound, self.obs_upper_bound)
            if not torch.equal(obs_clamp, obs):
                warnings.warn("Observation out of space!")
            obs = obs_clamp

        if self.clamp_action:
            action_clamp = action.clamp(self.action_lower_bound, self.action_upper_bound)
            if not torch.equal(action_clamp, action):
                warnings.warn("Action out of space!")
            action = action_clamp

        next_obs, reward, next_done, next_info = self.step(obs, action, info)

        if self.done_mask:
            next_obs = ~done * next_obs + done * obs
            reward = ~done * reward

        return next_obs, reward, next_done, next_info

    def step(self, obs: torch.Tensor, action: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        raise NotImplementedError

    def get_terminal_cost(self, obs: torch.Tensor) -> Union[torch.Tensor, None]:
        return None

    def get_constraint(self, obs: torch.Tensor) -> Union[torch.Tensor, None]:
        return None
