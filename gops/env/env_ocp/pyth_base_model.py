from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch

from gops.utils.gops_typing import InfoDict


class PythBaseModel(metaclass=ABCMeta):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 dt: float,
                 obs_lower_bound,
                 obs_upper_bound,
                 action_lower_bound,
                 action_upper_bound,
                 device: Union[torch.device, str, None] = None,
                 ):
        super(PythBaseModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt = dt
        self.obs_lower_bound = torch.tensor(obs_lower_bound, dtype=torch.float32, device=device)
        self.obs_upper_bound = torch.tensor(obs_upper_bound, dtype=torch.float32, device=device)
        self.action_lower_bound = torch.tensor(action_lower_bound, dtype=torch.float32, device=device)
        self.action_upper_bound = torch.tensor(action_upper_bound, dtype=torch.float32, device=device)
        self.device = device

    @abstractmethod
    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        pass

    def get_terminal_cost(self, obs: torch.Tensor) -> Union[torch.Tensor, None]:
        return None

    def get_constraint(self, obs: torch.Tensor) -> Union[torch.Tensor, None]:
        return None
