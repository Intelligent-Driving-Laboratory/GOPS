from typing import Tuple

import torch

from gops.env.env_wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict


class MaskAtDoneModel(ModelWrapper):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs, reward, next_done, next_info = super().forward(obs, action, done, info)
        done = done.bool()
        next_obs = ~done.unsqueeze(1) * next_obs + done.unsqueeze(1) * obs
        reward = ~done * reward
        return next_obs, reward, next_done, next_info
