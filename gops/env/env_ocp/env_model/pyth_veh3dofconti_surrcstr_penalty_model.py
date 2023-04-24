#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF model environment with surrounding vehicles constraint
#  Update: 2023-01-08, Jiaxin Gao: create environment

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


class Veh3dofcontiSurrCstrPenaltyModel(Veh3dofcontiModel):
    def __init__(
        self,
        pre_horizon: int,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        surr_veh_num: int = 1,
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

        state = info["state"]
        ref_points = info["ref_points"]
        path_num = info["path_num"]
        u_num = info["u_num"]
        t = info["ref_time"]
        surr_state = info["surr_state"]

        reward = self.compute_reward(obs, action, info)

        next_state = self.vehicle_dynamics.f_xu(state, action, self.dt)

        next_t = t + self.dt

        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]
        new_ref_point = torch.stack(
            (
                self.ref_traj.compute_x(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_y(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_phi(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_u(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
            ),
            dim=1,
        )
        next_ref_points[:, -1] = new_ref_point

        next_ego_obs = self.get_obs(next_state, next_ref_points)
        next_surr_state = self.surr_veh_model.forward(surr_state)
        sur_x_tf, sur_y_tf, sur_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                next_surr_state[..., 0], next_surr_state[..., 1], next_surr_state[..., 2],
            )
        sur_u_tf = next_surr_state[..., 3] - state[:, 3].unsqueeze(1)
        next_surr_obs = torch.stack((sur_x_tf, sur_y_tf, sur_phi_tf, sur_u_tf), 1).squeeze(2)
        next_obs = torch.cat((next_ego_obs, next_surr_obs), dim=1)


        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()
        next_info.update({
            "state": next_state,
            "ref_points": next_ref_points,
            "path_num": path_num,
            "u_num": u_num,
            "ref_time": next_t,
            "surr_state": next_surr_state,
            "constraint": self.get_constraint(next_obs, next_info),
        })

        next_done = self.judge_done(next_obs, next_info)

        return next_obs, reward, next_done, next_info

    def compute_reward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            info: InfoDict,
    ) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        v, w = obs[:, 4], obs[:, 5]
        steer, a_x = action[:, 0], action[:, 1]
        dis = - self.get_constraint(obs, info)
        collision_bound = 0.5
        dis_to_tanh = torch.maximum(8 - 8 * dis / collision_bound, torch.zeros_like(dis))
        punish_dis = torch.tanh(dis_to_tanh - 4) + 1

        return -(
                1.0 * delta_x ** 2
                + 1.0 * delta_y ** 2
                + 0.1 * delta_phi ** 2
                + 0.1 * delta_u ** 2
                + 0.5 * v ** 2
                + 0.5 * w ** 2
                + 0.5 * steer ** 2
                + 0.5 * a_x ** 2
                + 15.0 * punish_dis.squeeze()
        )
        # return -(
        #         0.5 * delta_x ** 2
        #         + 0.5 * delta_y ** 2
        #         + 0.1 * delta_phi ** 2
        #         + 0.1 * delta_u ** 2
        #         + 0.5 * v ** 2
        #         + 0.5 * w ** 2
        #         + 0.5 * steer ** 2
        #         + 0.5 * a_x ** 2
        #         + 15.0 * punish_dis.squeeze()
        # )


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
                        (surr_x + d * torch.cos(surr_phi)),
                        surr_y + d * torch.sin(surr_phi),
                    ),
                    dim=2,
                ),
                torch.cat(
                    (
                        (surr_x - d * torch.cos(surr_phi)),
                        surr_y - d * torch.sin(surr_phi),
                    ),
                    dim=2,
                ),
            ),
            dim=2,
        )

        min_dist = np.finfo(np.float32).max * torch.ones_like(surr_x).squeeze(-1)

        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of surrounding vehicles
                dist = torch.linalg.norm(
                    ego_center[:, i].unsqueeze(1) - surr_center[..., j, :], dim=2
                )

                min_dist = torch.minimum(
                    min_dist, dist
                )

        return 2 * r - min_dist

    def judge_done(self, obs: torch.Tensor, info: InfoDict) -> torch.Tensor:
        # delta_x, delta_y, delta_phi = obs[:, 0], obs[:, 1], obs[:, 2]
        # dis = - self.get_constraint(obs, info)
        # done = (
        #     (torch.abs(delta_x) > 5)
        #     | (torch.abs(delta_y) > 2)
        #     | (torch.abs(delta_phi) > np.pi)
        #     | (torch.any(dis < 0., dim=1))
        # )
        done = torch.zeros(obs.shape[0]).bool()
        return done

def ego_vehicle_coordinate_transform(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_phi: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1)
    cos_tf = torch.cos(-ego_phi)
    sin_tf = torch.sin(-ego_phi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_phi_tf = angle_normalize(ref_phi - ego_phi)
    return ref_x_tf, ref_y_tf, ref_phi_tf

def env_model_creator(**kwargs):
    return Veh3dofcontiSurrCstrPenaltyModel(**kwargs)
