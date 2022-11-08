import warnings
from typing import Tuple

import torch

from gops.env.env_wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict


class ClipActionModel(ModelWrapper):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        action_clip = action.clip(self.model.action_lower_bound, self.model.action_upper_bound)
        if not torch.equal(action_clip, action):
            warnings.warn("Action out of space!")

        return super().forward(obs, action_clip, done, info)
