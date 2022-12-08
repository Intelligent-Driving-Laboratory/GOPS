#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: model type environment wrapper that clips action to action space
#  Update: 2022-10-27, Yujie Yang: create action clip wrapper


import warnings
from typing import Tuple

import torch

from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict


class ClipActionModel(ModelWrapper):
    """
    Model type environment wrapper that clips action to action space.
    """

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        action_clip = action.clip(
            self.model.action_lower_bound, self.model.action_upper_bound
        )
        if not torch.equal(action_clip, action):
            warnings.warn("Action out of space!")

        return super().forward(obs, action_clip, done, info)
