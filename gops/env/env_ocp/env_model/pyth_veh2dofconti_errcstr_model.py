#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 2DOF model environment with tracking error constraint
#  Update: 2022-11-14, Yujie Yang: create environment

from typing import Any, Dict, Optional, Tuple, Union

import torch

from gops.env.env_ocp.env_model.pyth_veh2dofconti_model import Veh2dofcontiModel
from gops.utils.gops_typing import InfoDict


class Veh2dofcontiErrCstrModel(Veh2dofcontiModel):
    def __init__(
        self,
        pre_horizon: int,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        y_error_tol: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(pre_horizon, device, path_para, u_para)
        self.y_error_tol = y_error_tol

    # obs is o2 in data
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
        next_info["constraint"] = self.get_constraint(obs, info)
        return next_obs, reward, next_done, next_info

    def get_constraint(self, obs: torch.Tensor, info: InfoDict = None) -> torch.Tensor:
        y_error = obs[:, 0].unsqueeze(1)
        return y_error.abs() - self.y_error_tol


def env_model_creator(**kwargs):
    return Veh2dofcontiErrCstrModel(**kwargs)
