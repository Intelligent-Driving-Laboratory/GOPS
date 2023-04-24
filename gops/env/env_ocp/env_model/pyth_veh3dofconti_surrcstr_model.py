#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF model environment with surrounding vehicles constraint
#  Update: 2022-11-20, Yujie Yang: create environment

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_veh3dofconti_model import (
    VehicleDynamicsModel,
    Veh3dofcontiModel,
    angle_normalize,
)
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict


@dataclass
class SurrVehicleModel:
    # distance from front axle to rear axle
    l: float = 3.0
    dt: float = 0.1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x, y, phi, u, delta = state.split(1, dim=-1)
        next_x = x + u * torch.cos(phi) * self.dt
        next_y = y + u * torch.sin(phi) * self.dt
        next_phi = phi + u * torch.tan(delta) / self.l * self.dt
        next_phi = angle_normalize(next_phi)
        return torch.cat((next_x, next_y, next_phi, u, delta), dim=-1)


class Veh3dofcontiSurrCstrModel(Veh3dofcontiModel):
    def __init__(
        self,
        pre_horizon: int,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        surr_veh_num: int = 4,
        veh_length: float = 4.8,
        veh_width: float = 2.0,
        **kwargs: Any,
    ):
        self.state_dim = 6
        self.ego_obs_dim = 6
        self.ref_obs_dim = 4
        super(Veh3dofcontiModel, self).__init__(
            obs_dim=self.ego_obs_dim + self.ref_obs_dim * pre_horizon + surr_veh_num * 4,
            action_dim=2,
            dt=0.1,
            action_lower_bound=[-np.pi / 6, -3],
            action_upper_bound=[np.pi / 6, 3],
            device=device,
        )
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.ref_traj = MultiRefTrajModel(path_para, u_para)
        self.pre_horizon = pre_horizon

        self.surr_veh_model = SurrVehicleModel()
        self.surr_veh_num = surr_veh_num
        self.veh_length = veh_length
        self.veh_width = veh_width

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        ego_obs = obs[:, : self.ego_obs_dim + self.ref_obs_dim * self.pre_horizon]
        next_ego_obs, reward, next_done, next_info = super().forward(
            ego_obs, action, done, info
        )

        surr_state = info["surr_state"]
        next_surr_state = self.surr_veh_model.forward(surr_state)
        next_state = next_info["state"]
        next_surr_obs = next_surr_state[..., :4] - next_state[:, :4].unsqueeze(1)
        next_surr_obs = next_surr_obs.reshape((-1, self.surr_veh_num * 4))
        next_obs = torch.cat((next_ego_obs, next_surr_obs), dim=1)

        next_info.update({"surr_state": next_surr_state})
        next_info.update({"constraint": self.get_constraint(next_obs, next_info)})
                
        return next_obs, reward, next_done, next_info

    def get_constraint(self, obs: torch.Tensor, info: InfoDict) -> torch.Tensor:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.veh_width

        x, y, phi = info["state"][:, :3].split(1, dim=1)
        ego_center = torch.stack(
            (
                torch.cat((x + d * torch.cos(phi), y + d * torch.sin(phi)), dim=1),
                torch.cat((x - d * torch.cos(phi), y - d * torch.sin(phi)), dim=1),
            ),
            dim=1,
        )

        surr_x, surr_y, surr_phi = info["surr_state"][..., :3].split(1, dim=2)
        surr_center = torch.stack(
            (
                torch.cat(
                    (
                        surr_x + d * torch.cos(surr_phi),
                        surr_y + d * torch.sin(surr_phi),
                    ),
                    dim=2,
                ),
                torch.cat(
                    (
                        surr_x - d * torch.cos(surr_phi),
                        surr_y - d * torch.sin(surr_phi),
                    ),
                    dim=2,
                ),
            ),
            dim=2,
        )

        min_dist = np.finfo(np.float32).max * torch.ones_like(x)
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of surrounding vehicles
                dist = torch.linalg.norm(
                    ego_center[:, i].unsqueeze(1) - surr_center[..., j, :], dim=2
                )
                min_dist = torch.minimum(
                    min_dist, torch.min(dist, dim=1, keepdim=True).values
                )

        return 2 * r - min_dist


def env_model_creator(**kwargs):
    return Veh3dofcontiSurrCstrModel(**kwargs)
