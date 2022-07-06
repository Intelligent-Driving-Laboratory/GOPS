#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment


from gym import spaces
import gym
from gym.utils import seeding
from gops.env.pyth_veh2dofconti_model import VehicleDynamics, Veh2dofcontiModel
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


class SimuVeh2dofconti(gym.Env,):
    def __init__(self, num_future_data=0, num_agent=1, **kwargs):
        self.vehicle_dynamics = VehicleDynamics()
        self.obs = None
        self.veh_state = None
        self.veh_full_state = None
        self.action = None
        self.num_agent = num_agent
        self.done = np.zeros((self.num_agent,), dtype=np.int)
        self.base_frequency = 10
        self.interval_times = 200
        self.expected_vs = 20.
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (4)),
            high=np.array([np.inf] * (4)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.2 * np.pi / 9]),
                                           high=np.array([1.2 * np.pi / 9]),
                                           dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self, veh_state):
        v_ys, rs, delta_ys, delta_phis = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], veh_state[:, 3]
        lists_to_stack = [v_ys, rs, delta_ys, delta_phis]
        return np.stack(lists_to_stack, axis=1)

    def _get_state(self, obses):
        v_ys, rs, delta_ys, delta_phis = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3]
        lists_to_stack = [v_ys, rs, delta_ys, delta_phis]
        return np.stack(lists_to_stack, axis=1)

    def reset(self, **kwargs):
        init_x = np.random.uniform(0, 600, (self.num_agent,)).astype(np.float32)
        init_delta_y = np.random.normal(0, 1, (self.num_agent,)).astype(np.float32)
        init_y = self.vehicle_dynamics.path.compute_y(init_x, init_delta_y)
        init_delta_phi = np.random.normal(0, np.pi / 9, (self.num_agent,)).astype(np.float32)
        init_phi = self.vehicle_dynamics.path.compute_phi(init_x, init_delta_phi)
        init_v_x = 20.
        beta = np.random.normal(0, 0.15, (self.num_agent,)).astype(np.float32)
        init_v_y = init_v_x * np.tan(beta)
        init_r = np.random.normal(0, 0.3, (self.num_agent,)).astype(np.float32)
        init_veh_full_state = np.stack([init_v_y, init_r, init_y, init_phi], 1)
        self.veh_full_state = np.stack([init_v_y, init_r, init_y, init_phi, init_x], 1)
        self.veh_state = init_veh_full_state.copy()
        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(init_x), \
                           self.vehicle_dynamics.path.compute_path_phi(init_x)
        self.veh_state[:, 3] = self.veh_full_state[:, 3] - path_phi
        self.veh_state[:, 2] = self.veh_full_state[:, 2] - path_y
        self.obs = self._get_obs(self.veh_state)
        return self.obs

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm = action
        action = steer_norm * 1.2 * np.pi / 9
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self.action = action
        veh_state_tensor = torch.Tensor(self.veh_state)
        action_tensor = torch.from_numpy(self.action.astype("float32"))
        reward = self.vehicle_dynamics.compute_rewards(veh_state_tensor, action_tensor).numpy()
        self.veh_state, self.veh_full_state = self.vehicle_dynamics.simulation(veh_state_tensor, self.veh_full_state, action_tensor,
                                             base_freq=self.base_frequency)
        self.done = self.judge_done(self.veh_state)
        self.obs = self._get_obs(self.veh_state)
        info = {}
        return self.obs, reward, self.done, info

    def judge_done(self, veh_state):
        v_ys, rs, delta_ys, delta_phis = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], \
                                                   veh_state[:, 3]
        done = (np.abs(delta_ys) > 3) | (np.abs(delta_phis) > np.pi / 4.)

        return done

    def close(self):
        pass

    def render(self, mode='human'):
        pass

def env_creator(**kwargs):
    return SimuVeh2dofconti(**kwargs)

def scale_obs(obs):
    obs_scale = [1., 2., 1., 2.4]
    v_ys, rs, delta_ys, delta_phis = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    lists_to_stack = [v_ys * obs_scale[0], rs * obs_scale[1],
                      delta_ys * obs_scale[2], delta_phis * obs_scale[3]]
    return torch.stack(lists_to_stack, 1)

if __name__=="__main__":
    sys.path.append(r"G:\项目文档\gops开发相关\gops\gops\algorithm")
    base_dir = r"G:\项目文档\gops开发相关\gops\results\FHADP\220630-185721"
    # base_dir = r"E:\gops_更新基相关\gops_poly_modify_PG\results\FHADP\220620-065834-10A"
    net_dir = os.path.join(base_dir, r"apprfunc\apprfunc_{}.pkl".format(1999))
    parser = argparse.ArgumentParser()
    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_veh2dofconti")
    parser.add_argument("--algorithm", type=str, default="FHADP")
    parser.add_argument("--pre_horizon", type=int, default=30)
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=4, help="dim(State)")
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
    parser.add_argument("--value_func_type", type=str, default="POLY")
    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument('--policy_degree', type=int, default=2)
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    if policy_func_type == "POLY":
        pass
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument("--policy_hidden_activation", type=str, default="elu")
        parser.add_argument("--policy_output_activation", type=str, default="linear")
    # 3. Parameters for RL algorithm
    parser.add_argument("--policy_learning_rate", type=float, default=1e-5)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=5000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)

    if trainer_type == "off_serial_trainer":
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        parser.add_argument("--buffer_max_size", type=int, default=100000)
        parser.add_argument("--replay_batch_size", type=int, default=500)
        parser.add_argument("--sampler_sync_interval", type=int, default=1)
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
    env = SimuVeh2dofconti()
    model = Veh2dofcontiModel()
    obs = env.reset()
    env.obs = copy.deepcopy(obs)
    args = vars(parser.parse_args())
    args = init_args(env, **args)
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)
    networks.load_state_dict(torch.load(net_dir))
    v_x = []
    delta_phi = []
    delta_y = []
    x = []
    y_ref = []
    y = []
    phi = []
    reward_total = []

    obs = torch.from_numpy(obs.astype("float32"))
    model.reset(obs)
    for _ in range(200):
        batch_obs = scale_obs(obs)
        action = networks.policy(batch_obs)
        action = action.detach().numpy()
        obs, reward, done, info = env.step(action)
        obs = torch.from_numpy(obs.astype("float32"))
        x.append(env.veh_full_state[0, -1])
        path_y = env.vehicle_dynamics.path.compute_path_y(env.veh_full_state[0, -1])
        y_ref.append(path_y)
        y.append(env.veh_full_state[0, 2])

    plt.plot(x, y)
    plt.plot(x, y_ref, color='red')

    plt.show()