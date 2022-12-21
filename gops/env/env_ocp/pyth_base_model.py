#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: base class for pyth environments
#  Update: 2022-10-25, Yujie Yang: create base model
#  Update: 2022-10-27, Zhilong Zheng: redefine get_constraint and get_terminal_cost

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from gops.utils.gops_typing import InfoDict


class PythBaseModel(metaclass=ABCMeta):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        dt: Optional[float] = None,
        obs_lower_bound: Optional[Sequence] = None,
        obs_upper_bound: Optional[Sequence] = None,
        action_lower_bound: Optional[Sequence] = None,
        action_upper_bound: Optional[Sequence] = None,
        device: Union[torch.device, str, None] = None,
    ):
        super(PythBaseModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt = dt
        if obs_lower_bound is None:
            obs_lower_bound = [float("-inf")] * self.obs_dim
        if obs_upper_bound is None:
            obs_upper_bound = [float("inf")] * self.obs_dim
        if action_lower_bound is None:
            action_lower_bound = [float("-inf")] * self.action_dim
        if action_upper_bound is None:
            action_upper_bound = [float("inf")] * self.action_dim
        self.obs_lower_bound = torch.tensor(
            obs_lower_bound, dtype=torch.float32, device=device
        )
        self.obs_upper_bound = torch.tensor(
            obs_upper_bound, dtype=torch.float32, device=device
        )
        self.action_lower_bound = torch.tensor(
            action_lower_bound, dtype=torch.float32, device=device
        )
        self.action_upper_bound = torch.tensor(
            action_upper_bound, dtype=torch.float32, device=device
        )
        self.device = device

    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        pass

    # Define get_constraint as Callable
    # Trick for faster constraint evaluations
    # Subclass can realize it like:
    #   def get_constraint(self, obs: torch.Tensor, info: InfoDict) -> torch.Tensor:
    #       ...
    # This function should return Tensor of shape [n] (ndim = 1),
    # each element of which will be required to be lower than or equal to 0
    get_constraint: Callable[[torch.Tensor, InfoDict], torch.Tensor] = None

    # Just like get_constraint,
    # define function returning Tensor of shape [] (ndim = 0) in subclass
    # if you need
    get_terminal_cost: Callable[[torch.Tensor], torch.Tensor] = None

    @property
    def unwrapped(self):
        return self
