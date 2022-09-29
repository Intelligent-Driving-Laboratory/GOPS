import torch

import torch.nn as nn

from gops.utils.gops_typing import InfoDict


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(ModelWrapper, self).__init__()
        self.model = model

    # def __getattr__(self, name):
    #     if name.startswith("_"):
    #         raise AttributeError(f"attempted to get missing private attribute '{name}'")
    #     return getattr(self.model, name)

    def forward(self, state: torch.Tensor, action: torch.Tensor,info: InfoDict, beyond_done=None):
        return self.model.forward(state, action,info, beyond_done)

    # def forward_n_step(self, func, n, state: torch.Tensor):
    #     return self.model.forward_n_step(func, n, state)