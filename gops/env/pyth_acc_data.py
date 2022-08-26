#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Jiaxin Gao: create environment


from gym import spaces
import gym
from gym.utils import seeding
from gops.env.pyth_acc_model import PathAccModel
from gym.wrappers.time_limit import TimeLimit
from typing import Callable, Dict, List
import numpy as np
import copy
import time
import torch
import matplotlib.pyplot as plt
import argparse
import importlib
from gops.utils.init_args import init_args
import sys
import json
import os

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
        next_state = [delta_s * A[0, 0] + dalta_v * A[0, 1] + delta_a * A[0, 2] + B[0, 0] * actions,
                      delta_s * A[1, 0] + dalta_v * A[1, 1] + delta_a * A[1, 2] + B[1, 0] * actions,
                      delta_s * A[2, 0] + dalta_v * A[2, 1] + delta_a * A[2, 2] + B[2, 0] * actions]
        return torch.stack(next_state, 1)

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def simulation(self, obs, actions, base_freq):
        next_obs = self.prediction(obs, actions, base_freq)
        delta_s, dalta_v, delta_a = next_obs[:, 0], next_obs[:, 1], next_obs[:, 2]
        next_obs = torch.stack([delta_s, dalta_v, delta_a], 1)
        return next_obs

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        delta_s, dalta_v, delta_a = obs[:, 0], obs[:, 1], obs[:, 2]
        expect_a = actions
        devi_s = -torch.square(delta_s)
        devi_v = -torch.square(dalta_v)
        punish_a = -torch.square(delta_a)
        punish_action = -torch.square(expect_a)
        rewards = 0.4 * devi_s + 0.1 * devi_v + 0.2 * punish_a + 0.5 * punish_action
        return rewards

class PathAccData(gym.Env,):
    def __init__(self, num_agent=1, **kwargs):
        self.vehicle_dynamics = VehicleDynamics()
        self.obs = None
        self.action = None
        self.num_agent = num_agent
        self.done = np.zeros((self.num_agent,), dtype=np.int)
        self.base_frequency = 10
        self.interval_times = 200
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (3)),
            high=np.array([np.inf] * (3)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-2.]),
                                           high=np.array([1.]),
                                           dtype=np.float32)
        self.Max_step = 10
        self.cstep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        init_delta_s = np.random.normal(0, 1, (self.num_agent,)).astype(np.float32)
        init_delta_v = np.random.normal(0, 1, (self.num_agent,)).astype(np.float32)
        init_a = np.random.normal(-0.5, 0.5, (self.num_agent,)).astype(np.float32)
        self.obs = np.stack([init_delta_s, init_delta_v, init_a], 1)
        self.cstep = 0
        return self.obs[0]

    def _get_obs(self, veh_state):
        delta_s, delta_v, delta_a = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2]
        lists_to_stack = [delta_s, delta_v, delta_a]
        return np.stack(lists_to_stack, axis=1)

    def step(self, action):  # think of action is in range [-1, 1]
        self.action = action
        veh_state_tensor = torch.Tensor(self.obs)
        action_tensor = torch.from_numpy(self.action)
        reward = self.vehicle_dynamics.compute_rewards(veh_state_tensor, action_tensor).numpy()
        self.obs = self.vehicle_dynamics.simulation(veh_state_tensor, action_tensor,
                                             base_freq=self.base_frequency)
        self.obs = self._get_obs(self.obs)
        self.done = self.judge_done(self.obs)
        # if self.done:
        #     reward = reward - 10
        # else:
        #     reward = reward + 1
        info = {"TimeLimit.truncated": self.cstep > self.Max_step}
        self.cstep = self.cstep + 1
        return self.obs[0], reward, self.done, info

    def judge_done(self, veh_state):
        delta_s, dalta_v, delta_a = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2]
        done = (np.abs(delta_s) > 15) | (np.abs(dalta_v) > 3.)

        return done

    def close(self):
        pass

    def render(self, mode='human'):
        pass

def env_creator(**kwargs):
    return PathAccData(**kwargs)

# def scale_obs(obs):
#     obs_scale = [1., 2., 1., 2.4]
#     v_ys, rs, delta_ys, delta_phis = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
#     lists_to_stack = [v_ys * obs_scale[0], rs * obs_scale[1],
#                       delta_ys * obs_scale[2], delta_phis * obs_scale[3]]
#     return torch.stack(lists_to_stack, 1)

if __name__=="__main__":
    sys.path.append(r"E:\gops\gops\gops\algorithm")
    base_dir = r"E:\gops\gops\results\FHADP\220518-205609"
    net_dir = os.path.join(base_dir, r"apprfunc\apprfunc_{}.pkl".format(6000))
    parser = argparse.ArgumentParser()
    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_acc")
    parser.add_argument("--algorithm", type=str, default="FHADP")
    parser.add_argument("--pre_horizon", type=int, default=30)
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=3, help="dim(State)")
    parser.add_argument("--action_dim", type=int, default=1, help="dim(Action)")
    parser.add_argument("--action_high_limit", type=list, default=None)
    parser.add_argument("--action_low_limit", type=list, default=None)
    parser.add_argument(
        "--action_type", type=str, default="continu", help="Options: continu/discret"
    )
    parser.add_argument(
        "--is_render", type=bool, default=False, help="Draw environment animation"
    )
    parser.add_argument(
        "--is_adversary", type=bool, default=False, help="Adversary training"
    )
    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument("--value_func_type", type=str, default="MLP")
    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
        parser.add_argument("--policy_hidden_activation", type=str, default="elu")
        parser.add_argument("--policy_output_activation", type=str, default="linear")
    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--policy_learning_rate", type=float, default=3e-5)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=2000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={
            "mean": np.array([0], dtype=np.float32),
            "std": np.array([0.2], dtype=np.float32),
        },
    )

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)

    ################################################
    # 8. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=500)
    parser.add_argument("--log_save_interval", type=int, default=100)
    env = PathAccData()
    model = PathAccModel()
    obs = env.reset()
    obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
    args = vars(parser.parse_args())
    args = init_args(env, **args)
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)
    networks.load_state_dict(torch.load(net_dir))
    delta_s = []
    delta_v = []
    model.reset(obs)
    for _ in range(200):
        action = networks.policy(obs)
        obs, reward, done, constraint = model.forward(action)
        delta_s.append(obs[0, 0])
        delta_v.append(obs[0, 1])

    plt.plot(delta_s)
    plt.plot(delta_v)
    plt.show()