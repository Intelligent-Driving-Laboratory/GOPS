#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment
#  Update Date: 2022-04-20, Jiaxin Gao: modify veh2dof model
#  Update Date: 2022-09-22, Jiaxin Gao: add reference information

import numpy as np
import torch


class Veh2dofcontiModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics(**kwargs)
        self.base_frequency = 10.

    # obs is o2 in data
    def forward(self, obs: torch.Tensor, action: torch.Tensor, info: dict, beyond_done=torch.tensor(1)):
        steer_norm = action
        actions = steer_norm * 1.2 * np.pi / 9
        state = info["state"]
        ref_num = info["ref_num"]
        tc = info["t"]
        yc, phic, vc, wc = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        path_yc, path_phic = self.vehicle_dynamics.path.compute_path_y(tc, ref_num), \
                           self.vehicle_dynamics.path.compute_path_phi(tc, ref_num)
        obsc = torch.stack([yc - path_yc, phic - path_phic, vc, wc], 1)
        for i in range(self.vehicle_dynamics.prediction_horizon - 1):
            ref_y = self.vehicle_dynamics.path.compute_path_y(tc + (i + 1) / self.base_frequency, ref_num)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(tc + (i + 1) / self.base_frequency, ref_num)
            ref_obs = torch.stack([yc - ref_y, phic - ref_phi], 1)
            obsc = torch.hstack((obsc, ref_obs))
        reward = self.vehicle_dynamics.compute_rewards(obsc, actions)

        state_next = self.vehicle_dynamics.prediction(state, actions, self.base_frequency)
        y, phi, v, w = state_next[:, 0], state_next[:, 1], state_next[:, 2], state_next[:, 3]
        t = tc + 1 / self.base_frequency
        phi = torch.where(phi > np.pi, phi - 2 * np.pi, phi)
        phi = torch.where(phi <= -np.pi, phi + 2 * np.pi, phi)
        state_next = torch.stack([y, phi, v, w], 1)

        isdone = self.vehicle_dynamics.judge_done(state_next, ref_num, t)

        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(t, ref_num), \
                           self.vehicle_dynamics.path.compute_path_phi(t, ref_num)
        obs = torch.stack([y - path_y, phi - path_phi, v, w], 1)
        for i in range(self.vehicle_dynamics.prediction_horizon - 1):
            ref_y = self.vehicle_dynamics.path.compute_path_y(t + (i + 1) / self.base_frequency, ref_num)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(t + (i + 1) / self.base_frequency, ref_num)
            ref_obs = torch.stack([y - ref_y, phi - ref_phi], 1)
            obs = torch.hstack((obs, ref_obs))
        info["state"] = state_next
        info["constraint"] = None
        info["ref_num"] = info["ref_num"]
        info["t"] = t

        return obs, reward, isdone, info


class VehicleDynamics(object):
    def __init__(self, **kwargs):
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
        self.path = ReferencePath()
        self.prediction_horizon = kwargs["predictive_horizon"]

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
                    (m * v * u + delta_t * (l_f * k_f - l_r * k_r) * w - delta_t * k_f * steer * u - delta_t * m * np.square(u) * w) / (m * u - delta_t * (k_f + k_r)),
                    (I_z * w * u + delta_t * (l_r * k_f - l_r * k_r) * v - delta_t * l_f * k_f * steer * u) / (I_z * u - delta_t * (np.square(l_f) * k_f + np.square(l_r) * k_r))
                      ], dim=1)
        return next_state

    def judge_done(self, state, ref_num, t):
        y, phi, v, w = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        done = (torch.abs(y - self.path.compute_path_y(t, ref_num)) > 3) | \
               (torch.abs(phi - self.path.compute_path_phi(t, ref_num)) > np.pi / 4.)
        return done

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        delta_ys, delta_phis, v_ys, rs = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        devi_y = -torch.square(delta_ys)
        devi_phi = -torch.square(delta_phis)
        steers = actions[:, 0]
        punish_yaw_rate = -torch.square(rs)
        punish_steer = -torch.square(steers)
        punish_vys = - torch.square(v_ys)
        rewards = 2.0 * devi_y + 0.1 * devi_phi + 0.2 * punish_yaw_rate + 5 * punish_steer + 0.1 * punish_vys
        return rewards


class ReferencePath(object):
    def __init__(self):
        self.expect_v = 10.

    def compute_path_x(self, t, num):
        x = torch.where(num == 0, 10 * t + np.cos(2 * np.pi * t / 6), self.expect_v * t)
        return x

    def compute_path_y(self, t, num):
        y = torch.where(num == 0, 1.5 * torch.sin(2 * np.pi * t / 10),
                        torch.where(t < 5, torch.as_tensor(0.),
                                    torch.where(t < 9, 0.875 * t - 4.375,
                                    torch.where(t < 14, torch.as_tensor(3.5),
                                    torch.where(t < 18, -0.875 * t + 15.75, torch.as_tensor(0.))))))
        return y

    def compute_path_phi(self, t, num):
        phi = torch.where(num == 0, (1.5 * torch.sin(2 * torch.pi * (t + 0.001) / 10) - 1.5 * torch.sin(2 * torch.pi * t / 10))\
                  / (10 * t + torch.cos(2 * np.pi * (t + 0.001) / 6) - 10 * t + torch.cos(2 * np.pi * t / 6)),
                        torch.where(t <= 5, torch.as_tensor(0.),
                        torch.where(t <= 9, torch.as_tensor(((0.875 * (t + 0.001) - 4.375) - (0.875 * t - 4.375)) / (
                            self.expect_v * 0.001)),
                        torch.where(t <= 14, torch.as_tensor(0.),
                        torch.where(t <= 18, torch.as_tensor(((-0.875 * (t + 0.001) + 15.75) - (-0.875 * t + 15.75)) / (
                            self.expect_v * 0.001)), torch.as_tensor(0.))))))
        return torch.arctan(phi)


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh2dofconti`
    """
    return Veh2dofcontiModel(**kwargs)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result