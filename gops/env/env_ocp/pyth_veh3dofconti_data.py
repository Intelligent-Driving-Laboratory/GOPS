#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment


import gym
from gym.utils import seeding
import numpy as np
from gym.wrappers.time_limit import TimeLimit
from random import choice


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
        self.expected_vs = 10.
        self.path = ReferencePath()
        self.prediction_horizon = kwargs["predictive_horizon"]

    def f_xu(self, states, actions, delta_t):
        x, y, phi, u, v, w = states[0], states[1], states[2], \
                                             states[3], states[4], states[5]
        steer, a_x = actions[0], actions[1]
        k_f = self.vehicle_params['k_f']
        k_r = self.vehicle_params['k_r']
        l_f = self.vehicle_params['l_f']
        l_r = self.vehicle_params['l_r']
        m = self.vehicle_params['m']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']
        F_zf, F_zr = l_r * m * g / (l_f + l_r), l_f * m * g / (l_f + l_r)
        F_xf = np.where(a_x < 0, m * a_x / 2, np.zeros_like(a_x))
        F_xr = np.where(a_x < 0, m * a_x / 2, m * a_x)
        miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
        miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr
        alpha_f = np.arctan((v + l_f * w) / u) - steer
        alpha_r = np.arctan((v - l_r * r) / u)
        next_state = [x + delta_t * (u * np.cos(phi) - v * np.sin(phi)),
                      y + delta_t * (u * np.sin(phi) + v * np.cos(phi)),
                      phi + delta_t * w,
                      u + delta_t * (a_x + v * w),
                      (m * v * u + delta_t * (
                                  l_f * k_f - l_r * k_r) * r - delta_t * k_f * steer * u - delta_t * m * np.square(
                          u) * r) / (m * u - delta_t * (k_f + k_r)),
                      (-I_z * r * u - delta_t * (l_f * k_f - l_r * k_r) * v + delta_t * l_f * k_f * steer * u) / (
                                  delta_t * (np.square(l_f) * k_f + np.square(l_r) * k_r) - I_z * u)
                      ]
        alpha_f_bounds, alpha_r_bounds = 3 * miu_f * F_zf / k_f, 3 * miu_r * F_zr / k_r
        r_bounds = miu_r * g / np.abs(u)
        other = [alpha_f, alpha_r, next_state[2], alpha_f_bounds, alpha_r_bounds, r_bounds]
        return next_state, other


    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params

    def simulation(self, states, actions, base_freq, ref_num, t):
        state_next, others = self.prediction(states, actions, base_freq)
        x, y, phi, u, v, w = state_next[0], state_next[1], state_next[2], state_next[3], state_next[4], state_next[5]
        path_x, path_y, path_phi = self.path.compute_path_x(t, ref_num), \
                                   self.path.compute_path_y(t, ref_num), \
                           self.path.compute_path_phi(t, ref_num)
        obs = np.array([x - path_x, y - path_y, phi - path_phi, u - self.expected_vs, v, w], dtype=np.float32)
        for i in range(self.prediction_horizon - 1):
            ref_x = self.path.compute_path_x(t + (i + 1) / base_freq, ref_num)
            ref_y = self.path.compute_path_y(t + (i + 1) / base_freq, ref_num)
            ref_phi = self.path.compute_path_phi(t + (i + 1) / base_freq, ref_num)
            ref_obs = np.array([x - ref_x, y - ref_y, phi - ref_phi], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))

        if state_next[4] > np.pi:
            state_next[4] -= 2 * np.pi
        if state_next[4] <= -np.pi:
            state_next[4] += 2 * np.pi

        return state_next, obs, others

    def compute_rewards(self, obs, actions):  # obses and actions are tensors

        v_xs, v_ys, rs, delta_ys, delta_phis, xs = obs[0], obs[1], obs[2], \
                                                   obs[3], obs[4], obs[5]
        steers, a_xs = actions[0], actions[1]
        devi_v = -np.square(v_xs)
        devi_y = -np.square(delta_ys)
        devi_phi = -np.square(delta_phis)
        punish_yaw_rate = -np.square(rs)
        punish_steer = -np.square(steers)
        punish_a_x = -np.square(a_xs)
        punish_x = -np.square(xs)

        rewards = 0.05 * devi_v + 2.0 * devi_y + 0.05 * devi_phi + 0.05 * punish_yaw_rate + \
                  0.05 * punish_steer + 0.05 * punish_a_x + 0.02 * punish_x

        return rewards


class ReferencePath(object):
    def __init__(self):
        self.expect_v = 10

    def compute_path_x(self, t, num):
        x = np.zeros_like(t)
        if num == 0:
            x = 10 * t + np.cos(2 * np.pi * t / 6)
        elif num == 1:
            x = self.expect_v * t
        return x

    def compute_path_y(self, t, num):
        y = np.zeros_like(t)
        if num == 0:
            y = 1.5 * np.sin(2 * np.pi * t / 10)
        elif num == 1:
            if t <= 5:
                y = 0
            elif t <= 9:
                y = 0.875 * t - 4.375
            elif t <= 14:
                y = 3.5
            elif t <= 18:
                y = -0.875 * t + 15.75
            elif t > 18:
                y = 0
        return y

    def compute_path_phi(self, t, num):
        phi = np.zeros_like(t)
        if num == 0:
            phi = (1.5 * np.sin(2 * np.pi * (t + 0.001) / 10) - 1.5 * np.sin(2 * np.pi * t / 10)) \
                  / (10 * t + np.cos(2 * np.pi * (t + 0.001) / 6) - 10 * t + np.cos(2 * np.pi * t / 6))
        elif num == 1:
            if t <= 5:
                phi = 0
            elif t <= 9:
                phi = ((0.875 * (t + 0.001) - 4.375) - (0.875 * t - 4.375)) \
                      / (self.expect_v * 0.001)
            elif t <= 14:
                phi = 0
            elif t <= 18:
                phi = ((-0.875 * (t + 0.001) + 15.75) - (-0.875 * t + 15.75)) \
                      / (self.expect_v * 0.001)
            elif t > 18:
                phi = 0

        return np.arctan(phi)


class SimuVeh3dofconti(gym.Env,):
    def __init__(self, num_future_data=0, num_agent=1, **kwargs):

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)
        self.prediction_horizon = kwargs["predictive_horizon"]
        self.vehicle_dynamics = VehicleDynamics()
        self.num_agent = num_agent
        self.expected_vs = 10.
        self.base_frequency = 10
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (33)),
            high=np.array([np.inf] * (33)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-np.pi / 6, -3]),
                                           high=np.array([np.pi / 6, 3]),
                                           dtype=np.float32)
        self.obs = None
        self.state = None
        self.state_dim = 6
        self.ref_num = 1
        self.info_dict = {
            "state": {"shape": self.state_dim, "dtype": np.float32},
            "ref_num": {"shape": (), "dtype": np.uint8},
            "t": {"shape": (), "dtype": np.uint8},
        }
        self.seed()

    @property
    def additional_info(self):
        return self.info_dict

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        t = 20. * self.np_random.uniform(low=0., high=1.)
        flag = [0, 1]
        self.ref_num = self.np_random.choice(flag)
        path_x = self.vehicle_dynamics.path.compute_path_x(t)
        init_delta_x = self.np_random.normal(0, 2)
        init_x = path_x + init_delta_x
        init_delta_y = self.np_random.normal(0, 1)
        init_y = self.vehicle_dynamics.path.compute_path_y(t, self.ref_num) + init_delta_y
        init_delta_phi = self.np_random.normal(0, np.pi / 9)
        init_phi = self.vehicle_dynamics.path.compute_path_phi(t, self.ref_num) + init_delta_phi
        beta = self.np_random.normal(0, 0.15)
        init_r = self.np_random.normal(0, 0.3)
        init_v_x = self.np_random.uniform(low=5., high=15.)
        init_v_y = init_v_x * np.tan(beta)
        obs = np.array([init_v_x - self.expected_vs, init_v_y, init_r, init_delta_y, init_delta_phi, init_delta_x], dtype=np.float32)
        for i in range(self.prediction_horizon - 1):
            ref_x = self.vehicle_dynamics.path.compute_path_x(t + (i + 1) / self.base_frequency)
            ref_y = self.vehicle_dynamics.path.compute_path_y(t + (i + 1) / self.base_frequency, self.ref_num)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(t + (i + 1) / self.base_frequency, self.ref_num)
            ref_obs = np.array([init_x - ref_x, init_y - ref_y, init_phi - ref_phi], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))
        self.obs = obs
        self.state = np.array([init_v_x, init_v_y, init_r, init_y, init_phi, init_x, t], dtype=np.float32)
        return self.obs

    def step(self, action: np.ndarray, adv_action=None):  # think of action is in range [-1, 1]
        steer_norm, a_x_norm = action[0], action[1]
        action = np.stack([steer_norm * 1.2 * np.pi / 9, a_x_norm*3], 0)
        reward = self.vehicle_dynamics.compute_rewards(self.obs, action)
        self.state, self.obs, stability_related = self.vehicle_dynamics.simulation(self.state, action,
                                                                self.base_frequency, self.ref_num)
        self.done = self.judge_done(self.state, stability_related)

        state = np.array(self.state, dtype=np.float32)
        t = state[-1]
        x = state[5]
        x_ref = self.vehicle_dynamics.path.compute_path_x(t, self.ref_num)
        y = state[3]
        y_ref = self.vehicle_dynamics.path.compute_path_y(t, self.ref_num)
        info = {
            "state": state,
            "t": t,
            "x": x,
            "x_ref": x_ref,
            "y": y,
            "y_ref": y_ref,
            "ref_num": self.ref_num,
        }
        return self.obs, reward, self.done, info

    def judge_done(self, veh_state, stability_related):
        v_xs, v_ys, rs, ys, phis, xs, t = veh_state[0], veh_state[1], veh_state[2], \
                                                   veh_state[3], veh_state[4], veh_state[5], veh_state[6]
        alpha_f, alpha_r, r, alpha_f_bounds, alpha_r_bounds, r_bounds = stability_related[0], \
                                                                        stability_related[1], \
                                                                        stability_related[2], \
                                                                        stability_related[3], \
                                                                        stability_related[4], \
                                                                        stability_related[5]
        done = (np.abs(ys- self.vehicle_dynamics.path.compute_path_y(t, self.ref_num)) > 3) |\
               (np.abs(phis - self.vehicle_dynamics.path.compute_path_phi(t, self.ref_num)) > np.pi / 4.) |\
               (v_xs < 2)
               # (alpha_f < -alpha_f_bounds) | (alpha_f > alpha_f_bounds) | \
               # (alpha_r < -alpha_r_bounds) | (alpha_r > alpha_r_bounds) | \
               # (r < -r_bounds) | (r > r_bounds)
        return done

    def close(self):
        pass

    def render(self, mode='human'):
        pass


def env_creator(**kwargs):
    """
    make env `pyth_veh3dofconti`
    """
    return TimeLimit(SimuVeh3dofconti(**kwargs), 200)

if __name__ == "__main__":
    env = env_creator()
    env.seed()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        s, r, d, _ = env.step(action)
        print(s)
        # env.render()
        if d: env.reset()