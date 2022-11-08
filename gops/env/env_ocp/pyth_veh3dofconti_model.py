#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment
#  Update Date: 2022-04-20, Jiaxin Gao: modify veh3dof model


from typing import Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class Veh3dofcontiModel(PythBaseModel):
    def __init__(self,
                 pre_horizon: int,
                 device: Union[torch.device, str, None] = None,
                 ):
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.pre_horizon = pre_horizon

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
        ref_num = info["ref_num"]
        tc = info["ref_time"]
        xc, yc, phic, uc, vc, wc = state[:, 0], state[:, 1], state[:, 2], \
                                          state[:, 3], state[:, 4], state[:, 5]
        path_xc, path_yc, path_phic = self.vehicle_dynamics.compute_path_x(tc, ref_num), \
                                   self.vehicle_dynamics.compute_path_y(tc, ref_num), \
                                   self.vehicle_dynamics.compute_path_phi(tc, ref_num)
        obsc = torch.stack([xc - path_xc, yc - path_yc, phic - path_phic, uc, vc, wc], 1)
        for i in range(self.pre_horizon):
            ref_x = self.vehicle_dynamics.compute_path_x(tc + (i + 1) / self.base_frequency, ref_num)
            ref_y = self.vehicle_dynamics.compute_path_y(tc + (i + 1) / self.base_frequency, ref_num)
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
        isdone = self.vehicle_dynamics.judge_done(state_next, ref_num, t)
        path_x, path_y, path_phi = self.vehicle_dynamics.compute_path_x(t, ref_num),\
                                   self.vehicle_dynamics.compute_path_y(t, ref_num), \
                           self.vehicle_dynamics.compute_path_phi(t, ref_num)
        obs = torch.stack([x - path_x, y - path_y, phi - path_phi, u, v, w], 1)
        for i in range(self.pre_horizon):
            ref_x = self.vehicle_dynamics.compute_path_x(t + (i + 1) / self.base_frequency, ref_num)
            ref_y = self.vehicle_dynamics.compute_path_y(t + (i + 1) / self.base_frequency, ref_num)
            ref_obs = torch.stack([x - ref_x, y - ref_y], 1)
            obs = torch.hstack((obs, ref_obs))
        info["state"] = state_next
        info["constraint"] = None
        info["ref_num"] = info["ref_num"]
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
        self.vehicle_params.update(dict(F_zf=F_zf, F_zr=F_zr))

    def compute_path_x(self, t, num):
        x = torch.where(num == 0, 10 * t + torch.cos(2 * torch.pi * t / 6), self.vehicle_params['u'] * t)
        return x

    def compute_path_y(self, t, num):
        y = torch.where(num == 0, 1.5 * torch.sin(2 * torch.pi * t / 10),
                        torch.where(t < 5, torch.as_tensor(0.),
                                    torch.where(t < 9, 0.875 * t - 4.375,
                                                torch.where(t < 14, torch.as_tensor(3.5),
                                                            torch.where(t < 18, -0.875 * t + 15.75,
                                                                        torch.as_tensor(0.))))))
        return y

    def compute_path_phi(self, t, num):
        phi = torch.where(num == 0,
                          (1.5 * torch.sin(2 * torch.pi * (t + 0.001) / 10) - 1.5 * torch.sin(2 * torch.pi * t / 10)) \
                          / (10 * (t + 0.001) + torch.cos(2 * torch.pi * (t + 0.001) / 6) - 10 * t - torch.cos(2 * torch.pi * t / 6)),
                        torch.where(t <= 5, torch.as_tensor(0.),
                        torch.where(t <= 9, torch.as_tensor(((0.875 * (t + 0.001) - 4.375) - (0.875 * t - 4.375)) / (self.vehicle_params['u'] * 0.001)),
                        torch.where(t <= 14, torch.as_tensor(0.),
                        torch.where(t <= 18, torch.as_tensor(((-0.875 * (t + 0.001) + 15.75) - (-0.875 * t + 15.75)) / (self.vehicle_params['u'] * 0.001)),
                                                                          torch.as_tensor(0.))))))
        return torch.arctan(phi)

    def f_xu(self, states, actions, delta_t):
        x, y, phi, u, v, w = states[:, 0], states[:, 1], states[:, 2], \
                                             states[:, 3], states[:, 4], states[:, 5]
        steer, a_x = actions[:, 0], actions[:, 1]
        k_f = torch.tensor(self.vehicle_params['k_f'], dtype=torch.float32)
        k_r = torch.tensor(self.vehicle_params['k_r'], dtype=torch.float32)
        l_f = torch.tensor(self.vehicle_params['l_f'], dtype=torch.float32)
        l_r = torch.tensor(self.vehicle_params['l_r'], dtype=torch.float32)
        m = torch.tensor(self.vehicle_params['m'], dtype=torch.float32)
        I_z = torch.tensor(self.vehicle_params['I_z'], dtype=torch.float32)
        next_state = [x + delta_t * (u * torch.cos(phi) - v * torch.sin(phi)),
                      y + delta_t * (u * torch.sin(phi) + v * torch.cos(phi)),
                      phi + delta_t * w,
                      u + delta_t * a_x,
                      (m * v * u + delta_t * (
                                  l_f * k_f - l_r * k_r) * w - delta_t * k_f * steer * u - delta_t * m * torch.square(
                          u) * w) / (m * u - delta_t * (k_f + k_r)),
                      (I_z * w * u + delta_t * (l_f * k_f - l_r * k_r) * v - delta_t * l_f * k_f * steer * u) / (
                                  I_z * u - delta_t * (torch.square(l_f) * k_f + torch.square(l_r) * k_r)),
                      ]
        return torch.stack(next_state, 1)

    def prediction(self, x_1, u_1, frequency):
        state_next = self.f_xu(x_1, u_1, 1 / frequency)
        return state_next

    def judge_done(self, veh_state, ref_num, t):
        x, y, phi, u, v, w = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], \
                                                   veh_state[:, 3], veh_state[:, 4], veh_state[:, 5]
        done = (torch.abs(y - self.compute_path_y(t, ref_num)) > 2) |\
               (torch.abs(phi - self.compute_path_phi(t, ref_num)) > torch.pi / 4.) | \
               (torch.abs(x - self.compute_path_x(t, ref_num)) > 5)
        return done

    def compute_rewards(self, obs, actions):  # obses and actions are tensors

        delta_x, delta_y, delta_phi, u, v, w = obs[:, 0], obs[:, 1], obs[:, 2], \
                                                   obs[:, 3], obs[:, 4], obs[:, 5]
        steers, a_xs = actions[:, 0], actions[:, 1]
        devi_y = -torch.square(delta_y)
        devi_phi = -torch.square(delta_phi)
        punish_yaw_rate = -torch.square(w)
        punish_steer = -torch.square(steers)
        punish_a_x = -torch.square(a_xs)
        punish_x = -torch.square(delta_x)
        rewards = 0.1 * devi_y + 0.01 * devi_phi + 0.01 * punish_yaw_rate + \
                  0.01 * punish_steer + 0.01 * punish_a_x + 0.04 * punish_x

        return rewards


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh3dofconti`
    """

    return Veh3dofcontiModel(
        pre_horizon=kwargs["pre_horizon"],
        device=kwargs["device"],
    )
