#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment
#  Update Date: 2022-04-20, Jiaxin Gao: modify veh2dof model
#  Update Date: 2022-09-22, Jiaxin Gao: add reference information

from typing import Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class Veh2dofcontiModel(PythBaseModel):
    def __init__(self,
                 pre_horizon: int,
                 device: Union[torch.device, str, None] = None,
                 path_para: dict = None,
                 u_para: dict = None
                 ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.pre_horizon = pre_horizon
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
        info["constraint"] = None
        info["path_num"] = info["path_num"]
        info["ref_time"] = t

        return obs, reward, isdone, info


class VehicleDynamics(object):
    def __init__(self):
        self.vehicle_params = dict(k_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   k_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   l_f=1.06,  # distance from CG to front axle [m]
                                   l_r=1.85,  # distance from CG to rear axle [m]
                                   m=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   u=10.)
        l_f, l_r, mass, g = self.vehicle_params['l_f'], self.vehicle_params['l_r'], \
                            self.vehicle_params['m'], self.vehicle_params['g']
        F_zf, F_zr = l_r * mass * g / (l_f + l_r), l_f * mass * g / (l_f + l_r)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def inte_function(self, t, u_num, u_para):
        A = u_para['A']
        omega = u_para['omega']
        phi = u_para['phi']
        b = u_para['b']
        dis0 = 1 / omega * A * torch.sin(omega * t + phi) + b / 2 * t ** 2
        bool_0 = u_num == 0
        dis1 = u_para['speed'] * t
        bool_1 = u_num == 1
        dis = dis0 * bool_0 + dis1 * bool_1
        return dis

    def compute_path_u(self, t, u_num, u_para):
        u = np.zeros_like(t)
        A = u_para['A']
        omega = u_para['omega']
        phi = u_para['phi']
        b = u_para['b']

        bool_0 = u_num == 0
        u0 = A * torch.cos(omega * t + phi) + b * t
        bool_1 = u_num == 1
        u1 = u_para['speed'] * torch.ones_like(t)
        u = u0 * bool_0 + u1 * bool_1

        return u

    def compute_path_x(self, t, path_num, path_para, u_num, u_para):
        bool_0 = path_num == 0
        x0 = self.inte_function(t, u_num, u_para) * torch.ones_like(t)
        bool_1 = path_num == 1
        x1 = self.inte_function(t, u_num, u_para) * torch.ones_like(t)
        bool_2 = path_num == 2
        x2 = self.inte_function(t, u_num, u_para) * torch.ones_like(t)
        r = path_para['circle_radius']
        dis = self.inte_function(t, u_num, u_para)
        angle = dis / r
        x3 = r * torch.sin(angle)
        bool_3 = path_num == 3
        x = x0 * bool_0 + x1 * bool_1 + x2 * bool_2 + x3 * bool_3
        return x

    def compute_path_y(self, t, path_num, path_para, u_num, u_para):
        A = path_para['A_y']
        omega = path_para['omega_y']
        phi = path_para['phi_y']
        b = path_para['b_y']
        y0 = A * torch.sin(omega * t + phi) + b * t
        bool_0 = path_num == 0

        double_lane_control_point_1 = path_para['double_lane_control_point_1']
        double_lane_control_point_2 = path_para['double_lane_control_point_2']
        double_lane_control_point_3 = path_para['double_lane_control_point_3']
        double_lane_control_point_4 = path_para['double_lane_control_point_4']
        double_lane_y1 = path_para['double_lane_control_y1']
        double_lane_y3 = path_para['double_lane_control_y3']
        double_lane_y5 = path_para['double_lane_control_y5']
        double_lane_y2_a = path_para['double_lane_control_y2_a']
        double_lane_y2_b = path_para['double_lane_control_y2_b']
        double_lane_y4_a = path_para['double_lane_control_y4_a']
        double_lane_y4_b = path_para['double_lane_control_y4_b']
        y1 = torch.where(t < double_lane_control_point_1, double_lane_y1 * torch.ones_like(t),
                         torch.where(t < double_lane_control_point_2, double_lane_y2_a * t + double_lane_y2_b,
                                     torch.where(t < double_lane_control_point_3, double_lane_y3 * torch.ones_like(t),
                                                 torch.where(t < double_lane_control_point_4,
                                                             double_lane_y4_a * t + double_lane_y4_b,
                                                             double_lane_y5 * torch.ones_like(t)))))
        bool_1 = path_num == 1

        T = path_para['square_wave_period']
        A = path_para['square_wave_amplitude']
        x = self.compute_path_x(t, path_num, path_para, u_num, u_para)
        upper_int = (x / T).ceil()
        real_int = (x / T).round()
        y2 = torch.where(upper_int == real_int, - A * torch.ones_like(t), A * torch.ones_like(t))
        # y2 = - A * torch.ones_like(t)
        bool_2 = path_num == 2

        r = path_para['circle_radius']
        dis = self.inte_function(t, u_num, u_para)
        angle = dis / r
        y3 = r - r * torch.cos(angle)
        bool_3 = path_num == 3

        y = y0 * bool_0 + y1 * bool_1 + y2 * bool_2 + y3 * bool_3
        return y

    def compute_path_phi(self, t, path_num, path_para, u_num, u_para):
        phi0 = 0
        bool_0 = self.compute_path_y(t + 0.001, path_num, path_para, u_num, u_para) - self.compute_path_y(t, path_num,path_para,u_num,u_para) == 0
        phi1 = (self.compute_path_y(t + 0.001, path_num, path_para, u_num, u_para) - self.compute_path_y(t, path_num,path_para,u_num,u_para)) / (
                       self.compute_path_x(t + 0.001, path_num, path_para, u_num, u_para) - self.compute_path_x(t,path_num,path_para,u_num,u_para))
        bool_1 = self.compute_path_y(t + 0.001, path_num, path_para, u_num, u_para) - self.compute_path_y(t, path_num, path_para, u_num,u_para) != 0
        phi = phi0 * bool_0 + phi1 * bool_1
        return np.arctan(phi)

    def f_xu(self, states, actions, delta_t):
        y, phi, v, w = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        steer = actions[:, 0]
        u = torch.tensor(self.vehicle_params['u'], dtype=torch.float32)
        k_f = torch.tensor(self.vehicle_params['k_f'], dtype=torch.float32)
        k_r = torch.tensor(self.vehicle_params['k_r'], dtype=torch.float32)
        l_f = torch.tensor(self.vehicle_params['l_f'], dtype=torch.float32)
        l_r = torch.tensor(self.vehicle_params['l_r'], dtype=torch.float32)
        m = torch.tensor(self.vehicle_params['m'], dtype=torch.float32)
        I_z = torch.tensor(self.vehicle_params['I_z'], dtype=torch.float32)
        next_state = torch.stack([y + delta_t * (u * phi + v),
                      phi + delta_t * w,
                    (m * v * u + delta_t * (l_f * k_f - l_r * k_r) * w - delta_t * k_f * steer * u - delta_t * m * torch.square(u) * w) / (m * u - delta_t * (k_f + k_r)),
                    (I_z * w * u + delta_t * (l_f * k_f - l_r * k_r) * v - delta_t * l_f * k_f * steer * u) / (I_z * u - delta_t * (torch.square(l_f) * k_f + torch.square(l_r) * k_r))
                      ], dim=1)
        return next_state

    def judge_done(self, state, t, path_num, path_para, u_num, u_para):
        y, phi, v, w = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        done = (torch.abs(y - self.compute_path_y(t, path_num, path_para, u_num, u_para)) > 2) | \
               (torch.abs(phi - self.compute_path_phi(t, path_num, path_para, u_num, u_para)) > torch.pi / 4.)
        return done

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        delta_y, delta_phi, v, w = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        devi_y = -torch.square(delta_y)
        devi_phi = -torch.square(delta_phi)
        steers = actions[:, 0]
        punish_yaw_rate = -torch.square(w)
        punish_steer = -torch.square(steers)
        punish_vys = - torch.square(v)
        rewards = 0.1 * devi_y + 0.01 * devi_phi + 0.01 * punish_yaw_rate + 0.01 * punish_steer + 0.01 * punish_vys
        return rewards


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh2dofconti`
    """
    return Veh2dofcontiModel(
        pre_horizon=kwargs["pre_horizon"],
        device=kwargs["device"],
        path_para=None,
        u_para=None
    )
