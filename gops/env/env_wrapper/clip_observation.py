import warnings
from typing import Tuple

import torch

from gops.env.env_wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict


class ClipObservationModel(ModelWrapper):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs, reward, next_done, next_info = super().forward(obs, action, done, info)

        next_obs_clip = next_obs.clip(self.model.obs_lower_bound, self.model.obs_upper_bound)
        if not torch.equal(next_obs_clip, next_obs):
            warnings.warn("Observation out of space!")

        return next_obs_clip, reward, next_done, next_info
