#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF model environment
#  Update Date: 2022-04-20, Jiaxin Gao: create environment


from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_veh3dofconti import angle_normalize, VehicleDynamicsData
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict


class VehicleDynamicsModel(VehicleDynamicsData):
    def f_xu(self, states, actions, delta_t):
        x, y, phi, u, v, w = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )
        steer, a_x = actions[:, 0], actions[:, 1]
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
        next_state = [
            x + delta_t * (u * torch.cos(phi) - v * torch.sin(phi)),
            y + delta_t * (u * torch.sin(phi) + v * torch.cos(phi)),
            phi + delta_t * w,
            u + delta_t * a_x,
            (
                m * v * u
                + delta_t * (l_f * k_f - l_r * k_r) * w
                - delta_t * k_f * steer * u
                - delta_t * m * torch.square(u) * w
            )
            / (m * u - delta_t * (k_f + k_r)),
            (
                I_z * w * u
                + delta_t * (l_f * k_f - l_r * k_r) * v
                - delta_t * l_f * k_f * steer * u
            )
            / (I_z * u - delta_t * (l_f ** 2 * k_f + l_r ** 2 * k_r)),
        ]
        next_state[2] = angle_normalize(next_state[2])
        return torch.stack(next_state, 1)


class Veh3dofcontiModel(PythBaseModel):
    def __init__(
        self,
        pre_horizon: int = 10,
        device: Union[torch.device, str, None] = None,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        **kwargs,
    ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamicsModel()
        self.pre_horizon = pre_horizon
        ego_obs_dim = 6
        ref_obs_dim = 4
        super().__init__(
            obs_dim=ego_obs_dim + ref_obs_dim * pre_horizon,
            action_dim=2,
            dt=0.1,
            action_lower_bound=[-max_steer, -3],
            action_upper_bound=[max_steer, 3],
            device=device,
        )
        self.ref_traj = MultiRefTrajModel(path_para, u_para)

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

        reward = self.compute_reward(obs, action)

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

        next_obs = self.get_obs(next_state, next_ref_points)

        isdone = self.judge_done(next_obs)

        next_info = {}
        for key, value in info.items():
            next_info[key] = value.detach().clone()
        next_info.update({
            "state": next_state,
            "ref_points": next_ref_points,
            "path_num": path_num,
            "u_num": u_num,
            "ref_time": next_t,
        })
        return next_obs, reward, isdone, next_info

    def get_obs(self, state, ref_points):
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                state[:, 0], state[:, 1], state[:, 2],
                ref_points[..., 0], ref_points[..., 1], ref_points[..., 2],
            )
        ref_u_tf = ref_points[..., 3] - state[:, 3].unsqueeze(1)
        ego_obs = torch.concat((torch.stack(
            (ref_x_tf[:, 0], ref_y_tf[:, 0], ref_phi_tf[:, 0], ref_u_tf[:, 0]), dim=1),
            state[:, 4:]), dim=1)
        ref_obs = torch.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 2)[
            :, 1:].reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def compute_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        delta_x, delta_y, delta_phi, delta_u = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        w = obs[:, 5]
        steer, a_x = action[:, 0], action[:, 1]
        return -(
            0.04 * delta_x ** 2
            + 0.04 * delta_y ** 2
            + 0.02 * delta_phi ** 2
            + 0.02 * delta_u ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
            + 0.01 * a_x ** 2
        )

    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_x, delta_y, delta_phi = obs[:, 0], obs[:, 1], obs[:, 2]
        done = (
            (torch.abs(delta_x) > 10)
            | (torch.abs(delta_y) > 10)
            | (torch.abs(delta_phi) > np.pi)
        )
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
    """
    make env model `pyth_veh3dofconti`
    """
    return Veh3dofcontiModel(**kwargs)
