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
from gops.env.pyth_veh3dofconti_model import VehicleDynamics
from gym.wrappers.time_limit import TimeLimit
from typing import Callable, Dict, List
import numpy as np
import copy
import time
import torch
import matplotlib.pyplot as plt


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
        self.base_frequency = 200
        self.interval_times = 20
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (6 + self.num_future_data)),
            high=np.array([np.inf] * (6 + self.num_future_data)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.2 * np.pi / 9, -3]),
                                           high=np.array([1.2 * np.pi / 9, 3]),
                                           dtype=np.float32)

        # plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def _get_obs(self, veh_state, veh_full_state):
        future_delta_ys_list = []
        # future_delta_phi_list = []
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], \
                                                   veh_state[:, 3], veh_state[:, 4], veh_state[:, 5]

        v_xs, v_ys, rs, ys, phis, xs = veh_full_state[:, 0], veh_full_state[:, 1], veh_full_state[:, 2], \
                                       veh_full_state[:, 3], veh_full_state[:, 4], veh_full_state[:, 5]
        x_ = xs.copy()
        for _ in range(self.num_future_data):
            x_ += v_xs * 1. / self.base_frequency * self.interval_times * 2
            future_delta_ys_list.append(self.vehicle_dynamics.path.compute_delta_y(x_, ys))
            # future_delta_phi_list.append(self.vehicle_dynamics.path.compute_delta_phi(x_, phis))

        lists_to_stack = [v_xs - self.expected_vs, v_ys, rs, delta_ys, delta_phis, xs] + \
                         future_delta_ys_list  # + \
        # future_delta_phi_list
        return np.stack(lists_to_stack, axis=1)

    def _get_state(self, obses):
        delta_v_xs, v_ys, rs, delta_ys, delta_phis, xs = obses[:, 0], obses[:, 1], obses[:, 2], \
                                                         obses[:, 3], obses[:, 4], obses[:, 5]
        lists_to_stack = [delta_v_xs + self.expected_vs, v_ys, rs, delta_ys, delta_phis, xs]
        return np.stack(lists_to_stack, axis=1)

    def reset(self, **kwargs):
        if 'init_obs' in kwargs.keys():
            self.obs = kwargs.get('init_obs')
            self.veh_state = self._get_state(self.obs)
            init_x = self.veh_state[:, -1]
            path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(init_x), \
                               self.vehicle_dynamics.path.compute_path_phi(init_x)
            self.veh_full_state = self.veh_state.copy()
            self.veh_full_state[:, 4] = self.veh_state[:, 4] + path_phi
            self.veh_full_state[:, 3] = self.veh_state[:, 3] + path_y

            return self.obs

        self.simulation_time = 0
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
        if self.veh_full_state is None:
            self.veh_full_state = init_veh_full_state
        else:
            # for i, done in enumerate(self.done):
            #     self.veh_full_state[i, :] = np.where(done == 1, init_veh_full_state[i, :], self.veh_full_state[i, :])
            self.veh_full_state = np.where(self.done.reshape((-1, 1)) == 1, init_veh_full_state, self.veh_full_state)
        self.veh_state = self.veh_full_state.copy()
        path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(self.veh_full_state[:, -1]), \
                           self.vehicle_dynamics.path.compute_path_phi(self.veh_full_state[:, -1])
        self.veh_state[:, 4] = self.veh_full_state[:, 4] - path_phi
        self.veh_state[:, 3] = self.veh_full_state[:, 3] - path_y
        self.obs = self._get_obs(self.veh_state, self.veh_full_state)

        return self.obs

    def step(self, action):  # think of action is in range [-1, 1]
        steer_norm, a_x_norm = action[:, 0], action[:, 1]
        action = np.stack([steer_norm * 1.2 * np.pi / 9, a_x_norm * 3], 1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.simulation_time += self.interval_times * 1 / self.base_frequency
        self.action = action
        veh_state_tensor = torch.from_numpy(self.veh_state)
        action_tensor = torch.from_numpy(self.action)
        reward = self.vehicle_dynamics.compute_rewards(veh_state_tensor, action_tensor).numpy()
        self.veh_state, self.veh_full_state, stability_related = \
            self.vehicle_dynamics.simulation(self.veh_state, self.veh_full_state, self.action,
                                             base_freq=self.base_frequency, simu_times=self.interval_times)
        self.done = self.judge_done(self.veh_state, stability_related)
        self.obs = self._get_obs(self.veh_state, self.veh_full_state)
        info = {}
        return self.obs, reward, self.done, info

    def close(self):
        pass

    def judge_done(self, veh_state, stability_related):
        v_xs, v_ys, rs, delta_ys, delta_phis, xs = veh_state[:, 0], veh_state[:, 1], veh_state[:, 2], \
                                                   veh_state[:, 3], veh_state[:, 4], veh_state[:, 5]
        alpha_f, alpha_r, r, alpha_f_bounds, alpha_r_bounds, r_bounds = stability_related[:, 0], \
                                                                        stability_related[:, 1], \
                                                                        stability_related[:, 2], \
                                                                        stability_related[:, 3], \
                                                                        stability_related[:, 4], \
                                                                        stability_related[:, 5]
        done = (np.abs(delta_ys) > 3) | (np.abs(delta_phis) > np.pi / 4.) | (v_xs < 2) | \
               (alpha_f < -alpha_f_bounds) | (alpha_f > alpha_f_bounds) | \
               (alpha_r < -alpha_r_bounds) | (alpha_r > alpha_r_bounds) | \
               (r < -r_bounds) | (r > r_bounds)
        return done

    def render(self, mode='human'):
        pass
        # plt.cla()
        # v_x, v_y, r, delta_y, delta_phi, x = self.veh_state[0, 0], self.veh_state[0, 1], self.veh_state[0, 2], \
        #                                      self.veh_state[0, 3], self.veh_state[0, 4], self.veh_state[0, 5]
        # v_x, v_y, r, y, phi, x = self.veh_full_state[0, 0], self.veh_full_state[0, 1], \
        #                          self.veh_full_state[0, 2], self.veh_full_state[0, 3], \
        #                          self.veh_full_state[0, 4], self.veh_full_state[0, 5]
        # path_y, path_phi = self.vehicle_dynamics.path.compute_path_y(x), self.vehicle_dynamics.path.compute_path_phi(x)
        #
        # future_ys = self.obs[0, 6:]
        # xs = np.array(
        #     [x + i * v_x / self.base_frequency * self.interval_times * 2 for i in range(1, self.num_future_data + 1)])
        #
        # plt.plot(xs, -future_ys + y, 'r*')
        #
        # plt.title("Demo")
        # range_x, range_y = 100, 100
        # ax = plt.axes(xlim=(x - range_x / 2, x + range_x / 2),
        #               ylim=(-50, 50))
        # ax.add_patch(plt.Rectangle((x - range_x / 2, -50),
        #                            100, 100, edgecolor='black',
        #                            facecolor='none'))
        # plt.axis('equal')
        # plt.axis('off')
        # path_xs = np.linspace(x - range_x / 2, x + range_x / 2, 1000)
        # path_ys = self.vehicle_dynamics.path.compute_path_y(path_xs)
        # plt.plot(path_xs, path_ys)
        #
        # history_positions = list(self.history_positions)
        # history_xs = np.array(list(map(lambda x: x[0], history_positions)))
        # history_ys = np.array(list(map(lambda x: x[1], history_positions)))
        # plt.plot(history_xs, history_ys, 'g')
        #
        # def draw_rotate_rec(x, y, a, l, w, color='black'):
        #     RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
        #     RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
        #     LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
        #     LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
        #     plt.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
        #     plt.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
        #     plt.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
        #     plt.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)
        #
        # draw_rotate_rec(x, y, phi, 4.8, 2.2)
        # text_x, text_y_start = x - 20 - range_x / 2 - 20, 30
        # ge = iter(range(0, 1000, 4))
        # plt.text(text_x, text_y_start - next(ge), 'time: {:.2f}s'.format(self.simulation_time))
        # plt.text(text_x, text_y_start - next(ge), 'x: {:.2f}'.format(x))
        # plt.text(text_x, text_y_start - next(ge), 'y: {:.2f}'.format(y))
        # plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}'.format(path_y))
        # plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
        #
        # plt.text(text_x, text_y_start - next(ge), r'phi: {:.2f}rad (${:.2f}\degree$)'.format(phi, phi * 180 / np.pi, ))
        # plt.text(text_x, text_y_start - next(ge),
        #          r'path_phi: {:.2f}rad (${:.2f}\degree$)'.format(path_phi, path_phi * 180 / np.pi))
        # plt.text(text_x, text_y_start - next(ge),
        #          r'delta_phi: {:.2f}rad (${:.2f}\degree$)'.format(delta_phi, delta_phi * 180 / np.pi))
        #
        # plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(v_x))
        # plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.expected_vs))
        # plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(v_y))
        # plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(r))
        #
        # if self.action is not None:
        #     steer, a_x = self.action[0, 0], self.action[0, 1]
        #     plt.text(text_x, text_y_start - next(ge),
        #              r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
        #     plt.text(text_x, text_y_start - next(ge), r'a_x: {:.2f}m/s^2'.format(a_x))
        #
        # plt.axis([x - range_x / 2, x + range_x / 2, -range_y / 2, range_y / 2])
        #
        # plt.pause(0.001)
        # plt.show()


def env_creator(**kwargs):
    return SimuVeh3dofconti(**kwargs)
