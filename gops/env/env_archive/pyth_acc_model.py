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
import matplotlib.pyplot as plt

class PathAccModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        you need to define parameters here
        """
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obs = None
        self.actions = None

    def reset(self, obses):
        self.obs = obses
        self.actions = None

    def forward(self, actions: torch.Tensor):
        self.actions = actions
        rewards = self.vehicle_dynamics.compute_rewards(self.obs, actions)
        self.obs = self.vehicle_dynamics.prediction(self.obs, actions, self.base_frequency)
        mask = True
        return self.obs, rewards, mask, {"constraint": None}

    def forward_n_step(self, obs: torch.Tensor, func, n, done):
        done_list = []
        next_obs_list = []
        v_pi = torch.zeros((obs.shape[0],))
        self.reset(obs)
        for step in range(n):
            action = func(obs)
            obs, reward, done, constraint = self.forward(action)
            v_pi = v_pi + reward
            next_obs_list.append(obs)
            done_list.append(done)

        return next_obs_list, v_pi, done_list


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
                                   )

    def f_xu(self, obs, actions, tau):
        # c2d sample time T = 0.1 s
        A = np.array([[1., 0.1, 0.0133],
                      [0, 1., 0.0870],
                      [0, 0, 0.7515]])
        B = np.array([[0.0017], [0.0130], [0.2485]])
        A = torch.from_numpy(A.astype("float32"))
        B = torch.from_numpy(B.astype("float32"))
        delta_s, dalta_v, delta_a = obs[:, 0], obs[:, 1], obs[:, 2]
        next_state = [delta_s * A[0, 0] + dalta_v * A[0, 1] + delta_a * A[0, 2] + B[0, 0] * actions[:, 0],
                      delta_s * A[1, 0] + dalta_v * A[1, 1] + delta_a * A[1, 2] + B[1, 0] * actions[:, 0],
                      delta_s * A[2, 0] + dalta_v * A[2, 1] + delta_a * A[2, 2] + B[2, 0] * actions[:, 0]]

        return torch.stack(next_state, 1)

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        delta_s, dalta_v, delta_a = obs[:, 0], obs[:, 1], obs[:, 2]
        expect_a = actions
        devi_s = -torch.square(delta_s)
        devi_v = -torch.square(dalta_v)
        punish_a = -torch.square(delta_a)
        punish_action = -torch.square(expect_a)
        rewards = 0.4 * devi_s + 0.1 * devi_v + 0.2 * punish_a + 0.5 * punish_action
        return rewards


def env_model_creator(**kwargs):
    """
    make env model `pyth_acc`
    """
    return PathAccModel()


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
