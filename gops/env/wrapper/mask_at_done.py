#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: model type environment wrapper
#  Update: 2022-10-27, Yujie Yang: create mask done wrapper


from typing import Tuple

import torch

from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict


class MaskAtDoneModel(ModelWrapper):
    """
    Model type environment wrapper that masks observation and reward when done is True.
    """

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs, reward, next_done, next_info = super().forward(
            obs, action, done, info
        )
        done = done.bool()
        next_obs = ~done.unsqueeze(1) * next_obs + done.unsqueeze(1) * obs
        reward = ~done * reward
        next_done = next_done.bool() | done
        return next_obs, reward, next_done, next_info
