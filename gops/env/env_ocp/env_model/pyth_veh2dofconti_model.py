#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 2DOF model environment
#  Update Date: 2022-09-22, Jiaxin Gao: create environment

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_veh2dofconti import angle_normalize, VehicleDynamicsData
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel
from gops.utils.gops_typing import InfoDict


class VehicleDynamicsModel(VehicleDynamicsData):
    def f_xu(self, states, actions, delta_t):
        y, phi, v, w = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        steer = actions[:, 0]
        u = self.vehicle_params["u"]
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
        next_state = [
            y + delta_t * (u * phi + v),
            phi + delta_t * w,
            (
                m * v * u
                + delta_t * (l_f * k_f - l_r * k_r) * w
                - delta_t * k_f * steer * u
                - delta_t * m * u ** 2 * w
            )
            / (m * u - delta_t * (k_f + k_r)),
            (
                I_z * w * u
                + delta_t * (l_f * k_f - l_r * k_r) * v
                - delta_t * l_f * k_f * steer * u
            )
            / (I_z * u - delta_t * (l_f ** 2 * k_f + l_r ** 2 * k_r)),
        ]
        return torch.stack(next_state, 1)


class Veh2dofcontiModel(PythBaseModel):
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
        state_dim = 4
        super().__init__(
            obs_dim=state_dim + pre_horizon * 1,
            action_dim=1,
            dt=0.1,
            action_lower_bound=[-max_steer],
            action_upper_bound=[max_steer],
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

        # ground and ego vehicle coordinates change
        relative_state = state.clone()
        relative_state[:, :2] = 0
        next_relative_state = self.vehicle_dynamics.f_xu(
            relative_state, action, self.dt
        )
        y, phi = state[:, 0], state[:, 1]
        u = self.vehicle_dynamics.vehicle_params["u"]
        next_y = (
            y
            + u * torch.sin(phi) * self.dt
            + next_relative_state[:, 0] * torch.cos(phi)
        )
        next_phi = phi + next_relative_state[:, 1]
        next_phi = angle_normalize(next_phi)
        next_state = torch.cat(
            (torch.stack((next_y, next_phi), dim=1), next_relative_state[:, 2:]), dim=1
        )

        next_t = t + self.dt

        next_ref_points = ref_points.clone()
        next_ref_points[:, :-1] = ref_points[:, 1:]
        new_ref_point = torch.stack(
            (
                self.ref_traj.compute_y(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
                self.ref_traj.compute_phi(
                    next_t + self.pre_horizon * self.dt, path_num, u_num
                ),
            ),
            dim=1,
        )
        next_ref_points[:, -1] = new_ref_point

        ego_obs = torch.concat(
            (next_state[:, :2] - next_ref_points[:, 0], next_state[:, 2:]), dim=1
        )
        ref_obs = (next_state[:, :1].unsqueeze(1) - next_ref_points[:, 1:, :1]).reshape(
            (-1, self.pre_horizon)
        )
        next_obs = torch.concat((ego_obs, ref_obs), dim=1)

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

    def compute_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta_y, delta_phi, v, w = obs[:, :4].split(1, dim=1)
        steer = action
        return -(
            0.04 * delta_y ** 2
            + 0.02 * delta_phi ** 2
            + 0.01 * v ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
        ).squeeze(1)

    def judge_done(self, obs: torch.Tensor) -> torch.Tensor:
        delta_y, delta_phi = obs[:, :2].split(1, dim=1)
        return ((torch.abs(delta_y) > 2) | (torch.abs(delta_phi) > np.pi)).squeeze(1)


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh2dofconti`
    """
    return Veh2dofcontiModel(**kwargs)
