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
from gops.env.pyth_veh2dofconti_model import Veh2dofcontiModel
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
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.path = ReferencePath()

    def f_xu(self, states, actions, tau):
        A = np.array([[0.4411, -0.6398, 0, 0],
                      [0.0242, 0.2188, 0, 0],
                      [0.0703, 0.0171, 1, 2],
                      [0.0018, 0.0523, 0, 1]])
        B = np.array([[2.0350], [4.8124], [0.4046], [0.2952]])
        A = torch.from_numpy(A.astype("float32"))
        B = torch.from_numpy(B.astype("float32"))
        v_y, r, delta_y, delta_phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        next_state = [v_y * A[0, 0] + r * A[0, 1] + delta_y * A[0, 2] + delta_phi * A[0, 3] + B[0, 0] * actions,
                      v_y * A[1, 0] + r * A[1, 1] + delta_y * A[1, 2] + delta_phi * A[1, 3] + B[1, 0] * actions,
                      v_y * A[2, 0] + r * A[2, 1] + delta_y * A[2, 2] + delta_phi * A[2, 3] + B[2, 0] * actions,
                      v_y * A[3, 0] + r * A[3, 1] + delta_y * A[3, 2] + delta_phi * A[3, 3] + B[3, 0] * actions]
        return torch.stack(next_state, 1)

    def _get_obs(self, veh_states):
        v_ys, rs, delta_ys, delta_phis = veh_states[:, 0], veh_states[:, 1], veh_states[:, 2], veh_states[:, 3]
        lists_to_stack = [v_ys, rs, delta_ys, delta_phis]
        return torch.stack(lists_to_stack, 1)

    def _get_state(self, obses):
        v_ys, rs, delta_ys, delta_phis = obses[:, 0], obses[:, 1], obses[:, 2], obses[:, 3]
        lists_to_stack = [v_ys, rs, delta_ys, delta_phis]
        return torch.stack(lists_to_stack, 1)

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def simulation(self, states, actions, base_freq):
        next_states = self.prediction(states, actions, base_freq)
        v_ys, rs, delta_ys, delta_phis = next_states[:, 0], next_states[:, 1], next_states[:, 2], next_states[:, 3]
        delta_phis = torch.where(delta_phis > np.pi, delta_phis - 2 * np.pi, delta_phis)
        delta_phis = torch.where(delta_phis <= -np.pi, delta_phis + 2 * np.pi, delta_phis)
        next_states = torch.stack([v_ys, rs, delta_ys, delta_phis], 1)
        return next_states

    def compute_rewards(self, states, actions):  # obses and actions are tensors
        # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, xs
        # veh_full_state: v_xs, v_ys, rs, ys, phis, xs
        v_ys, rs, delta_ys, delta_phis = states[:, 0], states[:, 1], states[:, 2], \
                                                   states[:, 3]
        steers = actions
        devi_y = -torch.square(delta_ys)
        devi_phi = -torch.square(delta_phis)
        punish_yaw_rate = -torch.square(rs)
        punish_steer = -torch.square(steers)
        rewards = 0.4 * devi_y + 0.1 * devi_phi + 0.2 * punish_yaw_rate + 0.5 * punish_steer
        return rewards

class ReferencePath(object):
    def __init__(self):
        self.curve_list = [(7.5, 200., 0.), (2.5, 300., 0.), (-5., 400., 0.)]
        self.period = 1200.

    def compute_path_y(self, x):
        y = np.zeros_like(x, dtype=np.float32)
        for curve in self.curve_list:
            # 正弦的振幅，周期，平移
            # 这里是对3种正弦曲线进行了叠加。
            magnitude, T, shift = curve
            y += magnitude * np.sin((x - shift) * 2 * np.pi / T)
        return y

    def compute_path_phi(self, x):
        deriv = np.zeros_like(x, dtype=np.float32)
        for curve in self.curve_list:
            magnitude, T, shift = curve
            deriv += magnitude * 2 * np.pi / T * np.cos(
                (x - shift) * 2 * np.pi / T)
        return np.arctan(deriv)

    def compute_y(self, x, delta_y):
        y_ref = self.compute_path_y(x)
        return delta_y + y_ref

    def compute_delta_y(self, x, y):
        y_ref = self.compute_path_y(x)
        return y - y_ref

    def compute_phi(self, x, delta_phi):
        phi_ref = self.compute_path_phi(x)
        phi = delta_phi + phi_ref
        phi[phi > np.pi] -= 2 * np.pi
        phi[phi <= -np.pi] += 2 * np.pi
        return phi

    def compute_delta_phi(self, x, phi):
        phi_ref = self.compute_path_phi(x)
        delta_phi = phi - phi_ref
        delta_phi[delta_phi > np.pi] -= 2 * np.pi
        delta_phi[delta_phi <= -np.pi] += 2 * np.pi
        return delta_phi

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
        self.veh_full_state = init_veh_full_state
        self.veh_state = self.veh_full_state.copy()
        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(init_x), \
                           self.vehicle_dynamics.path.compute_path_phi(init_x)
        self.veh_state[:, 3] = self.veh_full_state[:, 3] - path_phi
        self.veh_state[:, 2] = self.veh_full_state[:, 2] - path_y
        self.obs = self._get_obs(self.veh_state)
        return self.obs[0]

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm = action
        action = steer_norm * 1.2 * np.pi / 9
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        self.action = action
        veh_state_tensor = torch.Tensor(self.veh_state)
        action_tensor = torch.from_numpy(self.action.astype("float32"))
        reward = self.vehicle_dynamics.compute_rewards(veh_state_tensor, action_tensor).numpy()
        self.veh_state = self.vehicle_dynamics.simulation(veh_state_tensor, action_tensor,
                                             base_freq=self.base_frequency)
        self.done = self.judge_done(self.veh_state)
        self.obs = self._get_obs(self.veh_state)
        info = {}
        return self.obs[0], reward, self.done, info

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
    sys.path.append(r"E:\gops\gops\gops\algorithm")
    base_dir = r"E:\gops\gops\results\FHADP\220517-125614"
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
    parser.add_argument("--value_func_type", type=str, default="MLP")
    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument("--policy_hidden_activation", type=str, default="elu")
        parser.add_argument("--policy_output_activation", type=str, default="tanh")
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
    env = SimuVeh2dofconti()
    model = Veh2dofcontiModel()
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
    v_x = []
    delta_phi = []
    delta_y = []
    reward_total = []
    model.reset(obs)
    for _ in range(200):
        batch_obs = scale_obs(obs)
        action = networks.policy(batch_obs)
        obs, reward, done, constraint = model.forward(action)
        reward_total.append(reward)
        delta_phi.append(obs[0, -1])
        delta_y.append(obs[0, 2])

    plt.plot(delta_phi)
    plt.plot(delta_y)
    plt.show()