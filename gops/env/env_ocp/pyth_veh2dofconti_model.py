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
    def __init__(self):
        super().__init__()
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.

    # obs is o2 in data
    def forward(self, obs: torch.Tensor, action: torch.Tensor, info: dict, beyond_done=torch.tensor(1)):
        steer_norm = action
        actions = steer_norm * 1.2 * np.pi / 9
        reward = self.vehicle_dynamics.compute_rewards(obs, actions)
        state = info["state"]
        state_next = self.vehicle_dynamics.prediction(state, actions, self.base_frequency)

        v_ys, rs, ys, phis, t = state_next[:, 0], state_next[:, 1], state_next[:, 2], state_next[:, 3], state_next[:, 4]
        phis = torch.where(phis > np.pi, phis - 2 * np.pi, phis)
        phis = torch.where(phis <= -np.pi, phis + 2 * np.pi, phis)
        state_next = torch.stack([v_ys, rs, ys, phis, t], 1)

        isdone = self.vehicle_dynamics.judge_done(state_next)

        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(t), \
                           self.vehicle_dynamics.path.compute_path_phi(t)
        obs = torch.stack([v_ys, rs, ys - path_y, phis - path_phi], 1)
        for i in range(self.vehicle_dynamics.prediction_horizon - 1):
            ref_y = self.vehicle_dynamics.path.compute_path_y(t + (i + 1) / self.base_frequency)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(t + (i + 1) / self.base_frequency)
            ref_obs = torch.stack([ys - ref_y, phis - ref_phi], 1)
            obs = torch.hstack((obs, ref_obs))
        info["state"] = state_next
        info["constraint"] = None

        return obs, reward, isdone, {"state":state_next}


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
                                   v_x=10.)
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.path = ReferencePath()
        self.prediction_horizon = 10

    def f_xu(self, states, actions, tau):
        v_y, r, delta_y, delta_phi, t = states[:, 0], states[:, 1], states[:, 2], \
                                             states[:, 3], states[:, 4]
        steer = actions[:, 0]
        v_x = torch.tensor(self.vehicle_params['v_x'], dtype=torch.float32)
        C_f = torch.tensor(self.vehicle_params['C_f'], dtype=torch.float32)
        C_r = torch.tensor(self.vehicle_params['C_r'], dtype=torch.float32)
        a = torch.tensor(self.vehicle_params['a'], dtype=torch.float32)
        b = torch.tensor(self.vehicle_params['b'], dtype=torch.float32)
        mass = torch.tensor(self.vehicle_params['mass'], dtype=torch.float32)
        I_z = torch.tensor(self.vehicle_params['I_z'], dtype=torch.float32)
        next_state = torch.stack([(mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * torch.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (torch.square(a) * C_f + torch.square(b) * C_r) - I_z * v_x),
                      delta_y + tau * (v_x * torch.sin(delta_phi) + v_y * torch.cos(delta_phi)),
                      delta_phi + tau * r, t + tau
                      ], dim=1)
        return next_state

    def judge_done(self, state):
        v_ys, rs, ys, phis, t = state[0], state[1], state[2], \
                                state[3], state[4]

        done = (np.abs(ys - self.path.compute_path_y(t)) > 3) | \
               (np.abs(phis - self.path.compute_path_phi(t)) > np.pi / 4.)
        return done

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        v_ys, rs, delta_ys, delta_phis = obs[:, 0], obs[:, 1], obs[:, 2], \
                                                   obs[:, 3]
        devi_y = -torch.square(delta_ys)
        devi_phi = -torch.square(delta_phis)
        steers = actions[:, 0]
        punish_yaw_rate = -torch.square(rs)
        punish_steer = -torch.square(steers)
        punish_vys = - torch.square(v_ys)
        rewards = 0.4 * devi_y + 0.1 * devi_phi + 0.2 * punish_yaw_rate + 0.5 * punish_steer + 0.1 * punish_vys
        return rewards


class ReferencePath(object):
    def __init__(self):
        self.expect_v = 10.
        self.period = 1200

    def compute_path_y(self, t):
        y = torch.sin((1 / 30) * self.expect_v * t)
        return y

    def compute_path_phi(self, t):
        phi = (torch.sin((1 / 30) * self.expect_v * (t + 0.001)) - torch.sin((1 / 30) * self.expect_v * t)) / (self.expect_v * 0.001)
        return torch.arctan(phi)


def env_model_creator(**kwargs):
    """
    make env model `pyth_veh2dofconti`
    """
    return Veh2dofcontiModel()


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