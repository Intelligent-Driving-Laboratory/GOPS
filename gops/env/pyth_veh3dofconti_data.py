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
from gops.env.pyth_veh3dofconti_model import Veh3dofcontiModel
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
                                   u=20
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.expected_vs = 20.
        self.path = ReferencePath()

    def f_xu(self, states, actions, tau):
        v_x, v_y, r, delta_y, delta_phi, x = states[0], states[1], states[2], \
                                             states[3], states[4], states[5]
        steer, a_x = actions[0], actions[1]
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
                      ]
        alpha_f_bounds, alpha_r_bounds = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bounds = miu_r * g / torch.abs(v_x)
        return torch.stack(next_state, 0), \
               torch.stack([alpha_f, alpha_r, next_state[2], alpha_f_bounds, alpha_r_bounds, r_bounds], 0)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params

    def simulation(self, states, full_states, actions, base_freq):
        # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, xs
        # veh_full_state: v_xs, v_ys, rs, ys, phis, xs
        # others: alpha_f, alpha_r, r, alpha_f_bounds, alpha_r_bounds, r_bounds
        # states = torch.from_numpy(states.copy())
        # actions = torch.tensor(actions)
        states, others = self.prediction(states, actions, base_freq)
        states = states.numpy()
        others = others.numpy()
        states[0] = np.clip(states[0], 1, 35)
        v_xs, v_ys, rs, phis = full_states[0], full_states[1], full_states[2], full_states[4]
        full_states[4] += rs / base_freq
        full_states[3] += (v_xs * np.sin(phis) + v_ys * np.cos(phis)) / base_freq
        full_states[-1] += (v_xs * np.cos(phis) - v_ys * np.sin(phis)) / base_freq
        full_states[0:3] = states[0:3].copy()
        path_y, path_phi = self.path.compute_path_y(full_states[-1]), \
                           self.path.compute_path_phi(full_states[-1])
        states[4] = full_states[4] - path_phi
        states[3] = full_states[3] - path_y
        if full_states[4] > np.pi:
            full_states[4] = full_states[4] - 2 * np.pi
        if full_states[4] <= -np.pi:
            full_states[4] = full_states[4] + 2 * np.pi
        if full_states[-1] > self.path.period:
            full_states[-1] = full_states[-1] - self.path.period
        if full_states[-1] <= 0:
            full_states[-1] = full_states[-1] + self.path.period
        states[-1] = full_states[-1]
        if states[4] > np.pi:
            states[4] = states[4] - 2 * np.pi
        if states[4] <= -np.pi:
            states[4] = states[4] + 2 * np.pi

        return states, full_states, others

    def compute_rewards(self, states, actions):  # obses and actions are tensors
        # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, xs
        # veh_full_state: v_xs, v_ys, rs, ys, phis, xs
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = states[0], states[1], states[2], \
                                                   states[3], states[4], states[5]
        steers, a_xs = actions[0], actions[1]
        devi_v = -torch.square(v_xs - self.expected_vs)
        devi_y = -torch.square(delta_ys)
        devi_phi = -torch.square(delta_phis)
        punish_yaw_rate = -torch.square(rs)
        punish_steer = -torch.square(steers)
        punish_a_x = -torch.square(a_xs)

        rewards = 0.1 * devi_v + 0.4 * devi_y + 1 * devi_phi + 0.2 * punish_yaw_rate + \
                  0.5 * punish_steer + 0.5 * punish_a_x

        return rewards


class ReferencePath(object):
    def __init__(self):
        self.curve_list = [(7.5, 200, 0.), (2.5, 300., 0.), (-5., 400., 0.)]
        self.period = 1200.

    def compute_path_y(self, x):
        y = np.zeros_like(x, dtype=np.float32)
        for curve in self.curve_list:
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


class SimuVeh3dofconti(gym.Env,):
    def __init__(self, num_future_data=0, num_agent=1, **kwargs):
        # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, xs
        # obs: delta_v_xs, v_ys, rs, delta_ys, delta_phis, xs, future_delta_ys1,..., future_delta_ysn,
        #         #      future_delta_phis1,..., future_delta_phisn
        # veh_full_state: v_xs, v_ys, rs, ys, phis, xs
        self.vehicle_dynamics = VehicleDynamics()
        self.num_future_data = num_future_data
        self.obs = None
        self.veh_state = None
        self.veh_full_state = None
        self.simulation_time = 0
        self.action = None
        self.num_agent = num_agent
        self.expected_vs = 20.
        self.done = np.zeros((self.num_agent,), dtype=np.int)
        self.base_frequency = 10
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (6 + self.num_future_data)),
            high=np.array([np.inf] * (6 + self.num_future_data)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.2 * np.pi / 9, -3]),
                                           high=np.array([1.2 * np.pi / 9, 3]),
                                           dtype=np.float32)
        self.Max_step = 100
        self.cstep = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def _get_obs(self, veh_state, veh_full_state):
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = veh_state[0], veh_state[1], veh_state[2], \
                                                   veh_state[3], veh_state[4], veh_state[5]

        v_xs, v_ys, rs, ys, phis, xs = veh_full_state[0], veh_full_state[1], veh_full_state[2], \
                                       veh_full_state[3], veh_full_state[4], veh_full_state[5]

        lists_to_stack = [v_xs - self.expected_vs, v_ys, rs, delta_ys, delta_phis, xs]

        return np.stack(lists_to_stack, axis=0)

    def _get_state(self, obses):
        delta_v_xs, v_ys, rs, delta_ys, delta_phis, xs = obses[0], obses[1], obses[2], \
                                                         obses[3], obses[4], obses[5]
        lists_to_stack = [delta_v_xs + self.expected_vs, v_ys, rs, delta_ys, delta_phis, xs]
        return np.stack(lists_to_stack, axis=0)

    def scale_obs(self, obs):
        obs_scale = [1., 1., 2., 1., 2.4, 1 / 1200]
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = obs[0], obs[1], obs[2], \
                                                   obs[3], obs[4], obs[5]
        lists_to_stack = [v_xs * obs_scale[0], v_ys * obs_scale[1], rs * obs_scale[2],
                          delta_ys * obs_scale[3], delta_phis * obs_scale[4], xs * obs_scale[5]]
        return lists_to_stack

    def reset(self, **kwargs):
        init_x = np.random.uniform(0, 600, (self.num_agent,)).astype(np.float32)
        init_delta_y = np.random.normal(0, 1, (self.num_agent,)).astype(np.float32)
        init_y = self.vehicle_dynamics.path.compute_y(init_x, init_delta_y)
        init_delta_phi = np.random.normal(0, np.pi / 9, (self.num_agent,)).astype(np.float32)
        init_phi = self.vehicle_dynamics.path.compute_phi(init_x, init_delta_phi)
        init_v_x = np.random.uniform(15, 25, (self.num_agent,)).astype(np.float32)
        beta = np.random.normal(0, 0.15, (self.num_agent,)).astype(np.float32)
        init_v_y = init_v_x * np.tan(beta)
        init_r = np.random.normal(0, 0.3, (self.num_agent,)).astype(np.float32)
        init_veh_full_state = np.stack([init_v_x, init_v_y, init_r, init_y, init_phi, init_x], 1)
        self.veh_full_state = init_veh_full_state[0]
        self.veh_state = self.veh_full_state.copy()
        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(self.veh_full_state[-1]), \
                           self.vehicle_dynamics.path.compute_path_phi(self.veh_full_state[-1])
        self.veh_state[4] = self.veh_full_state[4] - path_phi
        self.veh_state[3] = self.veh_full_state[3] - path_y
        self.obs = self._get_obs(self.veh_state, self.veh_full_state)
        self.cstep = 0
        return self.scale_obs(self.obs)

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm, a_x_norm = action[0], action[1]
        action = np.stack([steer_norm * 1.2 * np.pi / 9, a_x_norm*3], 0)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.action = action
        veh_state_tensor = torch.from_numpy(self.veh_state)
        action_tensor = torch.from_numpy(self.action)
        reward = self.vehicle_dynamics.compute_rewards(veh_state_tensor, action_tensor).numpy()
        self.veh_state, self.veh_full_state, stability_related = \
            self.vehicle_dynamics.simulation(veh_state_tensor, self.veh_full_state, action_tensor,
                                             base_freq=self.base_frequency)
        self.done = self.judge_done(self.veh_state, stability_related)
        if self.done:
            pass
        else:
            reward = reward + 1
        self.obs = self._get_obs(self.veh_state, self.veh_full_state)
        info = {"TimeLimit.truncated": self.cstep > self.Max_step}
        self.cstep = self.cstep + 1
        return self.scale_obs(self.obs), reward, self.done, info

    def judge_done(self, veh_state, stability_related):
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = veh_state[0], veh_state[1], veh_state[2], \
                                                   veh_state[3], veh_state[4], veh_state[5]
        alpha_f, alpha_r, r, alpha_f_bounds, alpha_r_bounds, r_bounds = stability_related[0], \
                                                                        stability_related[1], \
                                                                        stability_related[2], \
                                                                        stability_related[3], \
                                                                        stability_related[4], \
                                                                        stability_related[5]
        done = (np.abs(delta_ys) > 3) | (np.abs(delta_phis) > np.pi / 4.) | (v_xs < 2) | \
               (alpha_f < -alpha_f_bounds) | (alpha_f > alpha_f_bounds) | \
               (alpha_r < -alpha_r_bounds) | (alpha_r > alpha_r_bounds) | \
               (r < -r_bounds) | (r > r_bounds)
        return done

    def close(self):
        pass

    def render(self, mode='human'):
        pass


def env_creator(**kwargs):
    return SimuVeh3dofconti(**kwargs)


def unscale_obs(obs):
    obs_scale = [1., 1., 2., 1., 2.4, 1 / 1200]
    v_xs, v_ys, rs, delta_ys, delta_phis, xs = obs[:, 0], obs[:, 1], obs[:, 2], \
                                               obs[:, 3], obs[:, 4], obs[:, 5]
    lists_to_stack = [v_xs / obs_scale[0], v_ys / obs_scale[1], rs / obs_scale[2],
                      delta_ys / obs_scale[3], delta_phis / obs_scale[4], xs / obs_scale[5]]
    return torch.stack(lists_to_stack, 1)


if __name__=="__main__":
    sys.path.append(r"G:\项目文档\gops开发相关\gops\gops\algorithm")
    base_dir = r"G:\项目文档\gops开发相关\gops\results\FHADP\220907-151803"
    net_dir = os.path.join(base_dir, r"apprfunc\apprfunc_{}.pkl".format(1999))
    parser = argparse.ArgumentParser()
    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_veh3dofconti")
    parser.add_argument("--algorithm", type=str, default="FHADP")
    parser.add_argument("--pre_horizon", type=int, default=30)
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=6, help="dim(State)")
    parser.add_argument("--action_dim", type=int, default=2, help="dim(Action)")
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
    parser.add_argument("--num_eval_episode", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100)

    ################################################
    # 8. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=50)
    parser.add_argument("--log_save_interval", type=int, default=100)
    env = SimuVeh3dofconti()
    model = Veh3dofcontiModel()
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
    x = []
    y = []
    y_ref = []
    phi = []
    reward_total = []
    model.reset(unscale_obs(obs))
    for _ in range(300):
        action = networks.policy(obs)
        action = action.detach()[0].numpy()
        obs, reward, done, info = env.step(action)
        obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        x.append(env.veh_full_state[-1])
        path_y = env.vehicle_dynamics.path.compute_path_y(env.veh_full_state[-1])
        y_ref.append(path_y)
        y.append(env.veh_full_state[3])

    plt.plot(x, y)
    plt.plot(x, y_ref, color='red')
    plt.show()