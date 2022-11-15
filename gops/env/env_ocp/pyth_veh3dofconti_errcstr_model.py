#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF model environment with tracking error constraint


from typing import Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.env.env_ocp.pyth_veh3dofconti_model import VehicleDynamics
from gops.utils.gops_typing import InfoDict


class Veh3dofcontiErrCstrModel(PythBaseModel):
    def __init__(self,
                 pre_horizon: int,
                 device: Union[torch.device, str, None] = None,
                 path_para:dict = None,
                 u_para:dict = None,
                 y_error_tol: float = 0.2,
                 u_error_tol: float = 2.0,
                 **kwargs,
                 ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.pre_horizon = pre_horizon
        self.y_error_tol = y_error_tol
        self.u_error_tol = u_error_tol
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
        path_value = [1., 2 * np.pi / 6, 0, 10, 1.5, 2 * np.pi / 10, 0, 0, 5, 9, 14, 18, 0, 3.5, 0, 0.875, -4.375, -0.875, 15.75, 5, 1, 200]
        self.path_para = dict(zip(path_key, path_value))
        if path_para != None:
            for i in path_para.keys(): self.path_para[i] = path_para[i]

        u_key = ['A', 'omega', 'phi', 'b', 'speed']

        u_value = [1, 2 * np.pi / 6, 0, 0.5, 5]


        self.u_para = dict(zip(u_key, u_value))

        if u_para != None:
            for i in u_para.keys(): self.u_para[i] = u_para[i]


        state_dim = 6
        super().__init__(
            obs_dim=state_dim + pre_horizon * 2,
            action_dim=2,
            dt=1 / self.base_frequency,
            action_lower_bound=[-np.pi / 6, -3],
            action_upper_bound=[np.pi / 6, 3],
            device=device,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor, done: torch.Tensor, info: InfoDict) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        steer_norm, a_xs_norm = action[:, 0], action[:, 1]
        actions = torch.stack([steer_norm, a_xs_norm], 1)
        state = info["state"]
        path_num = info["path_num"]
        u_num = info["u_num"]
        tc = info["ref_time"]


        xc, yc, phic, uc, vc, wc = state[:, 0], state[:, 1], state[:, 2], \
                                          state[:, 3], state[:, 4], state[:, 5]
        path_xc, path_yc, path_phic = self.vehicle_dynamics.compute_path_x(tc, path_num, self.path_para, u_num, self.u_para), \
                                   self.vehicle_dynamics.compute_path_y(tc, path_num, self.path_para, u_num, self.u_para), \
                                   self.vehicle_dynamics.compute_path_phi(tc, path_num, self.path_para, u_num, self.u_para)
        path_uc = self.vehicle_dynamics.compute_path_u(tc, u_num, self.u_para)
        obsc = torch.stack([xc - path_xc, yc - path_yc, phic - path_phic, uc - path_uc, vc, wc], 1)
        for i in range(self.pre_horizon):
            ref_x = self.vehicle_dynamics.compute_path_x(tc + (i + 1) / self.base_frequency, path_num, self.path_para, u_num, self.u_para)
            ref_y = self.vehicle_dynamics.compute_path_y(tc + (i + 1) / self.base_frequency, path_num, self.path_para, u_num, self.u_para)
            ref_obs = torch.stack([xc - ref_x, yc - ref_y], 1)
            obsc = torch.hstack((obsc, ref_obs))
        reward = self.vehicle_dynamics.compute_rewards(obsc, actions)
        state_next = self.vehicle_dynamics.prediction(state, actions,
                                                              self.base_frequency)
        x, y, phi, u, v, w = state_next[:, 0], state_next[:, 1], state_next[:, 2], \
                                                   state_next[:, 3], state_next[:, 4], state_next[:, 5]
        t = tc + 1 / self.base_frequency
        phi = torch.where(phi > torch.pi, phi - 2 * torch.pi, phi)
        phi = torch.where(phi <= -torch.pi, phi + 2 * torch.pi, phi)
        state_next = torch.stack([x, y, phi, u, v, w], 1)
        isdone = self.vehicle_dynamics.judge_done(state_next, t, path_num, self.path_para, u_num, self.u_para)
        path_x, path_y, path_phi = self.vehicle_dynamics.compute_path_x(t, path_num, self.path_para, u_num, self.u_para),\
                                   self.vehicle_dynamics.compute_path_y(t, path_num, self.path_para, u_num, self.u_para), \
                           self.vehicle_dynamics.compute_path_phi(t, path_num, self.path_para, u_num, self.u_para)
        path_u = self.vehicle_dynamics.compute_path_u(t, u_num, self.u_para)
        obs = torch.stack([x - path_x, y - path_y, phi - path_phi, u - path_u, v, w], 1)
        for i in range(self.pre_horizon):
            ref_x = self.vehicle_dynamics.compute_path_x(t + (i + 1) / self.base_frequency, path_num, self.path_para, u_num, self.u_para)
            ref_y = self.vehicle_dynamics.compute_path_y(t + (i + 1) / self.base_frequency, path_num, self.path_para, u_num, self.u_para)
            ref_obs = torch.stack([x - ref_x, y - ref_y], 1)
            obs = torch.hstack((obs, ref_obs))
        info["state"] = state_next
        info["constraint"] = None
        info["path_num"] = info["path_num"]
        info["ref_time"] = t
        return obs, reward, isdone, info

    def get_constraint(self, obs: torch.Tensor) -> torch.Tensor:
        y_error = obs[:, 1].unsqueeze(1)
        u_error = obs[:, 3].unsqueeze(1)
        # TODO: there is inconsistency in whether obs[3] is u or u_error
        constraint = torch.stack((y_error.abs() - self.y_error_tol, u_error.abs() - self.u_error_tol), dim=1)
        return constraint


def env_model_creator(**kwargs):
    return Veh3dofcontiErrCstrModel(**kwargs)
