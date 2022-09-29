#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment
#  Update Date: 2022-04-20, Jiaxin Gao: modify veh3dof model


import math
import warnings
import numpy as np
import torch
import copy
from gym.wrappers.time_limit import TimeLimit
import gym

class Veh3dofcontiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.expected_vs = 10.


    def forward(self, obs: torch.Tensor, action: torch.Tensor, info: dict, beyond_done=torch.tensor(1)):
        steer_norm, a_xs_norm = action[:, 0], action[:, 1]
        actions = torch.stack([steer_norm * 1.2 * np.pi / 9, a_xs_norm * 3.], 1)
        state = info["state"]
        ref_num = info["ref_num"]
        v_xc, v_yc, rc, yc, phic, xc, tc = state[:, 0], state[:, 1], state[:, 2], \
                                          state[:, 3], state[:, 4], state[:, 5], state[:, 6]
        path_xc, path_yc, path_phic = self.vehicle_dynamics.path.compute_path_x(tc), \
                                   self.vehicle_dynamics.path.compute_path_y(tc, ref_num), \
                                   self.vehicle_dynamics.path.compute_path_phi(tc, ref_num)
        obsc = torch.stack([v_xc - self.expected_vs, v_yc, rc, yc - path_yc, phic - path_phic, xc - path_xc], 1)
        for i in range(self.vehicle_dynamics.prediction_horizon - 1):
            ref_x = self.vehicle_dynamics.path.compute_path_x(tc + (i + 1) / self.base_frequency)
            ref_y = self.vehicle_dynamics.path.compute_path_y(tc + (i + 1) / self.base_frequency, ref_num)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(tc + (i + 1) / self.base_frequency, ref_num)
            ref_obs = torch.stack([xc - ref_x, yc - ref_y, phic - ref_phi], 1)
            obsc = torch.hstack((obsc, ref_obs))
        reward = self.vehicle_dynamics.compute_rewards(obsc, actions)
        state_next, stability_related = self.vehicle_dynamics.prediction(state, actions,
                                                              self.base_frequency)
        v_xs, v_ys, rs, ys, phis, xs, t = state_next[:, 0], state_next[:, 1], state_next[:, 2], \
                                                   state_next[:, 3], state_next[:, 4], state_next[:, 5], state_next[:, 6]
        v_xs = clip_by_tensor(v_xs, 1, 35)
        phis = torch.where(phis > np.pi, phis - 2 * np.pi, phis)
        phis = torch.where(phis <= -np.pi, phis + 2 * np.pi, phis)
        state_next = torch.stack([v_xs, v_ys, rs, ys, phis, xs, t], 1)

        isdone = self.vehicle_dynamics.judge_done(state_next, stability_related, ref_num)
        path_x, path_y, path_phi = self.vehicle_dynamics.path.compute_path_x(t),\
                                   self.vehicle_dynamics.path.compute_path_y(t, ref_num), \
                           self.vehicle_dynamics.path.compute_path_phi(t, ref_num)
        obs = torch.stack([v_xs - self.expected_vs, v_ys, rs, ys - path_y, phis - path_phi, xs - path_x], 1)
        for i in range(self.vehicle_dynamics.prediction_horizon - 1):
            ref_x = self.vehicle_dynamics.path.compute_path_x(t + (i + 1) / self.base_frequency)
            ref_y = self.vehicle_dynamics.path.compute_path_y(t + (i + 1) / self.base_frequency, ref_num)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(t + (i + 1) / self.base_frequency, ref_num)
            ref_obs = torch.stack([xs - ref_x, ys - ref_y, phis - ref_phi], 1)
            obs = torch.hstack((obs, ref_obs))
        info["state"] = state_next
        info["constraint"] = None
        info["ref_num"] = info["ref_num"]

        return obs, reward, isdone, info



class VehicleDynamics(object):
    def __init__(self):
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   u=10
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.expected_vs = 10.
        self.prediction_horizon = 10
        self.path = ReferencePath()

    def f_xu(self, states, actions, tau):
        v_x, v_y, r, delta_y, delta_phi, x, t = states[:, 0], states[:, 1], states[:, 2], \
                                             states[:, 3], states[:, 4], states[:, 5], states[:, 6]
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = torch.tensor(self.vehicle_params['C_f'], dtype=torch.float32)
        C_r = torch.tensor(self.vehicle_params['C_r'], dtype=torch.float32)
        a = torch.tensor(self.vehicle_params['a'], dtype=torch.float32)
        b = torch.tensor(self.vehicle_params['b'], dtype=torch.float32)
        mass = torch.tensor(self.vehicle_params['mass'], dtype=torch.float32)
        I_z = torch.tensor(self.vehicle_params['I_z'], dtype=torch.float32)
        miu = torch.tensor(self.vehicle_params['miu'], dtype=torch.float32)
        g = torch.tensor(self.vehicle_params['g'], dtype=torch.float32)
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = torch.where(a_x < 0, mass * a_x / 2, torch.zeros_like(a_x))
        F_xr = torch.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = torch.sqrt(torch.square(miu * F_zf) - torch.square(F_xf)) / F_zf
        miu_r = torch.sqrt(torch.square(miu * F_zr) - torch.square(F_xr)) / F_zr
        alpha_f = torch.atan((v_y + a * r) / v_x) - steer
        alpha_r = torch.atan((v_y - b * r) / v_x)
        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                                  a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * torch.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                                  tau * (torch.square(a) * C_f + torch.square(b) * C_r) - I_z * v_x),
                      delta_y + tau * (v_x * torch.sin(delta_phi) + v_y * torch.cos(delta_phi)),
                      delta_phi + tau * r,
                      x + tau * (v_x * torch.cos(delta_phi) - v_y * torch.sin(delta_phi)),
                      t + tau
                      ]
        alpha_f_bounds, alpha_r_bounds = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bounds = miu_r * g / torch.abs(v_x)
        return torch.stack(next_state, 1), \
               torch.stack([alpha_f, alpha_r, next_state[2], alpha_f_bounds, alpha_r_bounds, r_bounds], 1)

    def prediction(self, x_1, u_1, frequency):
        state_next, others = self.f_xu(x_1, u_1, 1 / frequency)
        return state_next, others

    def judge_done(self, veh_state, stability_related, ref_num):
        v_xs, v_ys, rs, ys, phis, xs, t = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], \
                                                   veh_state[:, 3], veh_state[:, 4], veh_state[:, 5], veh_state[:, 6]
        alpha_f, alpha_r, r, alpha_f_bounds, alpha_r_bounds, r_bounds = stability_related[:, 0], \
                                                                        stability_related[:, 1], \
                                                                        stability_related[:, 2], \
                                                                        stability_related[:, 3], \
                                                                        stability_related[:, 4], \
                                                                        stability_related[:, 5]
        done = (torch.abs(ys - self.path.compute_path_y(t, ref_num)) > 3) | (torch.abs(phis - self.path.compute_path_phi(t, ref_num)) > np.pi / 4.) | \
               (v_xs < 2)
               # (alpha_f < -alpha_f_bounds) | (alpha_f > alpha_f_bounds) | \
               # (alpha_r < -alpha_r_bounds) | (alpha_r > alpha_r_bounds) | \
               # (r < -r_bounds) | (r > r_bounds)
        return done


    def compute_rewards(self, obs, actions):  # obses and actions are tensors

        v_xs, v_ys, rs, delta_ys, delta_phis, xs = obs[:, 0], obs[:, 1], obs[:, 2], \
                                                   obs[:, 3], obs[:, 4], obs[:, 5]
        steers, a_xs = actions[:, 0], actions[:, 1]
        devi_v = -torch.square(v_xs)
        devi_y = -torch.square(delta_ys)
        devi_phi = -torch.square(delta_phis)
        punish_yaw_rate = -torch.square(rs)
        punish_steer = -torch.square(steers)
        punish_a_x = -torch.square(a_xs)
        punish_x = -torch.square(xs)

        rewards = 0.05 * devi_v + 2.0 * devi_y + 0.05 * devi_phi + 0.05 * punish_yaw_rate + \
                  0.05 * punish_steer + 0.05 * punish_a_x + 0.02 * punish_x

        return rewards


class ReferencePath(object):
    def __init__(self):
        self.expect_v = 10.

    def compute_path_x(self, t, num):
        if num ==0:
            x = self.expect_v * t+ torch.cos(2*np.pi/6*t)
        elif num==1:
            x = self.expect_v * t
        x_1 = self.expect_v * t+ torch.cos(2*np.pi/6*t)
        x_2 = self.expect_v * t
        x = (num==0)*x_1 + (num==1)*x_2
        return

    def compute_path_y(self, t, num):
        y = torch.where(num == 0, torch.sin((1 / 30) * self.expect_v * t),
                        torch.where(t < (50 / self.expect_v), torch.as_tensor(0.),
                                    torch.where(t < (90 / self.expect_v), 0.0875 * self.expect_v * t - 4.375,
                                    torch.where(t < (140 / self.expect_v), torch.as_tensor(3.5),
                                    torch.where(t < (180 / self.expect_v), -0.0875 * self.expect_v * t + 15.75, torch.as_tensor(0.))))))
        return y

    def compute_path_phi(self, t, num):
        phi = torch.where(num == 0, (torch.sin((1 / 30) * self.expect_v * (t + 0.001)) - torch.sin((1 / 30) * self.expect_v * t)) / (self.expect_v * 0.001),
                        torch.where(t < (50 / self.expect_v), torch.as_tensor(0.),
                        torch.where(t < (90 / self.expect_v), torch.as_tensor(((0.0875 * self.expect_v * (t + 0.001) - 4.375) - (0.0875 * self.expect_v * t - 4.375)) / (
                            self.expect_v * 0.001)),
                        torch.where(t < (140 / self.expect_v), torch.as_tensor(0.),
                        torch.where(t < (180 / self.expect_v), torch.as_tensor(((-0.0875 * self.expect_v * (t + 0.001) + 15.75) - (-0.0875 * self.expect_v * t + 15.75)) / (
                            self.expect_v * 0.001)), torch.as_tensor(0.))))))
        return torch.arctan(phi)


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh3dofconti`
    """

    return Veh3dofcontiModel()


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
