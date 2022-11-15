#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 2DOF model environment with tracking error constraint

from typing import Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_veh2dofconti_model import VehicleDynamics
from gops.utils.gops_typing import InfoDict


class Veh2dofcontiErrCstrModel(PythBaseModel):
    def __init__(self,
                 pre_horizon: int,
                 device: Union[torch.device, str, None] = None,
                 path_para: dict = None,
                 u_para: dict = None,
                 y_error_tol: float = 0.2,
                 **kwargs,
                 ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.pre_horizon = pre_horizon
        self.y_error_tol = y_error_tol
        path_key = ['A_x',
                    'omega_x',
                    'phi_x',
                    'b_x',
                    'A_y',
                    'omega_y',
                    'phi_y',
                    'b_y',
                    'double_lane_control_point_1',
                    'double_lane_control_point_2',
                    'double_lane_control_point_3',
                    'double_lane_control_point_4',
                    'double_lane_control_y1',
                    'double_lane_control_y3',
                    'double_lane_control_y5',
                    'double_lane_control_y2_a',
                    'double_lane_control_y2_b',
                    'double_lane_control_y4_a',
                    'double_lane_control_y4_b',
                    'square_wave_period',
                    'square_wave_amplitude',
                    'circle_radius',
                    ]
        path_value = [1., 2 * np.pi / 6, 0, 10, 1.5, 2 * np.pi / 10, 0, 0, 5, 9, 14, 18, 0, 3.5, 0, 0.875, -4.375,
                      -0.875, 15.75, 5, 1, 200]
        self.path_para = dict(zip(path_key, path_value))
        if path_para != None:
            for i in path_para.keys(): self.path_para[i] = path_para[i]

        u_key = ['A', 'omega', 'phi', 'b', 'speed']

        u_value = [1, 2 * np.pi / 6, 0, 0.5, 5]

        self.u_para = dict(zip(u_key, u_value))

        if u_para != None:
            for i in u_para.keys(): self.u_para[i] = u_para[i]

        state_dim = 4
        super().__init__(
            obs_dim=state_dim + pre_horizon * 1,
            action_dim=1,
            dt=1 / self.base_frequency,
            action_lower_bound=[-np.pi / 6],
            action_upper_bound=[np.pi / 6],
            device=device,
        )

    # obs is o2 in data
    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        steer_norm = action
        actions = steer_norm
        state = info["state"]
        path_num = info["path_num"]
        u_num = info["u_num"]
        tc = info["ref_time"]
        yc, phic, vc, wc = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        path_yc, path_phic = self.vehicle_dynamics.compute_path_y(tc, path_num, self.path_para, u_num, self.u_para), \
                           self.vehicle_dynamics.compute_path_phi(tc, path_num, self.path_para, u_num, self.u_para)
        obsc = torch.stack([yc - path_yc, phic - path_phic, vc, wc], 1)
        for i in range(self.pre_horizon):
            ref_y = self.vehicle_dynamics.compute_path_y(tc + (i + 1) / self.base_frequency, path_num, self.path_para, u_num, self.u_para)
            ref_obs = torch.stack([yc - ref_y], 1)
            obsc = torch.hstack((obsc, ref_obs))
        reward = self.vehicle_dynamics.compute_rewards(obsc, actions)

        y_current, phi_current, v_current, w_current = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        y_relative = torch.zeros_like(y_current)
        phi_relative = torch.zeros_like(phi_current)
        relative_state = torch.stack([y_relative, phi_relative, v_current, w_current], 1)
        relative_state_next = self.vehicle_dynamics.prediction(relative_state, actions, self.base_frequency)
        delta_y, delta_phi, v_next, w_next = relative_state_next[:, 0], relative_state_next[:, 1], relative_state_next[:, 2], relative_state_next[:, 3]
        y, phi, v, w = y_current + self.vehicle_dynamics.compute_path_u(tc, u_num, self.u_para) / self.base_frequency * torch.sin(phi_current) + delta_y * torch.cos(phi_current), phi_current + delta_phi, v_next, w_current

        t = tc + 1 / self.base_frequency
        phi = torch.where(phi > torch.pi, phi - 2 * torch.pi, phi)
        phi = torch.where(phi <= -torch.pi, phi + 2 * torch.pi, phi)
        state_next = torch.stack([y, phi, v, w], 1)

        isdone = self.vehicle_dynamics.judge_done(state_next, t, path_num, self.path_para, u_num, self.u_para)

        path_y, path_phi = self.vehicle_dynamics.compute_path_y(t, path_num, self.path_para, u_num, self.u_para), \
                           self.vehicle_dynamics.compute_path_phi(t, path_num, self.path_para, u_num, self.u_para)
        obs = torch.stack([y - path_y, phi - path_phi, v, w], 1)
        for i in range(self.pre_horizon):
            ref_y = self.vehicle_dynamics.compute_path_y(t + (i + 1) / self.base_frequency, path_num, self.path_para, u_num, self.u_para)
            ref_obs = torch.stack([y - ref_y], 1)
            obs = torch.hstack((obs, ref_obs))
        info["state"] = state_next
        info["constraint"] = self.get_constraint(obs)
        info["path_num"] = info["path_num"]
        info["ref_time"] = t

        return obs, reward, isdone, info

    def get_constraint(self, obs: torch.Tensor) -> torch.Tensor:
        y_error = obs[:, 0].unsqueeze(1)
        return y_error.abs() - self.y_error_tol


def env_model_creator(**kwargs):
    return Veh2dofcontiErrCstrModel(**kwargs)
