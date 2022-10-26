from typing import Tuple

import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class ModelWrapper:
    def __init__(self, model: PythBaseModel):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        return self.model.forward(obs, action, done, info)
