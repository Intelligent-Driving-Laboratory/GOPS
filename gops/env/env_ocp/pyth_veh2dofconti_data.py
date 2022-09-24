#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 2DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment
#  Update Date: 2022-09-21, Jiaxin Gao: change to tracking problem

import gym
from gym.utils import seeding
import numpy as np
from gym.wrappers.time_limit import TimeLimit

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
        v_y, r, delta_y, delta_phi, t = states[0], states[1], states[2], \
                                             states[3], states[4]
        steer = actions[0]
        v_x = self.vehicle_params['v_x']
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        next_state = np.stack([(mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * np.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (np.square(a) * C_f + np.square(b) * C_r) - I_z * v_x),
                      delta_y + tau * (v_x * np.sin(delta_phi) + v_y * np.cos(delta_phi)),
                      delta_phi + tau * r, t + tau
                      ])
        return next_state

    def judge_done(self, state):
        done = True
        return done

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def simulation(self, state, action, base_freq):
        state_next = self.prediction(state, action, base_freq)
        v_y, r, y, phi, t = state_next[0], state_next[1], state_next[2], state_next[3], state_next[4]
        path_y, path_phi = self.path.compute_path_y(t), \
                           self.path.compute_path_phi(t)
        obs = np.array([v_y, r, y - path_y, phi - path_phi], dtype=np.float32)
        for i in range(self.prediction_horizon - 1):
            ref_y = self.path.compute_path_y(t + (i + 1) / base_freq)
            ref_phi = self.path.compute_path_phi(t + (i + 1) / base_freq)
            ref_obs = np.array([y - ref_y, phi - ref_phi], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))
        if state_next[3] > np.pi:
            state_next[3] -= 2 * np.pi
        if state_next[3] <= -np.pi:
            state_next[3] += 2 * np.pi
        return state_next, obs

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        v_ys, rs, delta_ys, delta_phis = obs[0], obs[1], obs[2], \
                                                   obs[3]
        devi_y = -np.square(delta_ys)
        devi_phi = -np.square(delta_phis)
        steers = actions[0]
        punish_yaw_rate = -np.square(rs)
        punish_steer = -np.square(steers)
        punish_vys = - np.square(v_ys)
        rewards = 0.4 * devi_y + 0.1 * devi_phi + 0.2 * punish_yaw_rate + 0.5 * punish_steer + 0.1 * punish_vys
        return rewards


class ReferencePath(object):
    def __init__(self):
        self.expect_v = 10
        self.period = 1200

    def compute_path_y(self, t):
        y = np.sin((1 / 30) * self.expect_v * t)
        return y

    def compute_path_phi(self, t):
        phi = (np.sin((1 / 30) * self.expect_v * (t + 0.001)) - np.sin((1 / 30) * self.expect_v * t)) / (self.expect_v * 0.001)
        return np.arctan(phi)


class SimuVeh2dofconti(gym.Env,):
    def __init__(self, num_future_data=0, num_agent=1, **kwargs):
        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)
        self.prediction_horizon = 10
        self.vehicle_dynamics = VehicleDynamics()
        self.num_agent = num_agent
        self.base_frequency = 10
        self.expected_vs = 10.
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (22)),
            high=np.array([np.inf] * (22)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.2 * np.pi / 9]),
                                           high=np.array([1.2 * np.pi / 9]),
                                           dtype=np.float32)
        self.obs = None
        self.state = None
        self.state_dim = 5
        self.info_dict = {"state":{"shape": self.state_dim, "dtype": np.float32}}
        self.seed()

    @property
    def additional_info(self):
        return self.info_dict

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        t = 60. * self.np_random.uniform(low=0., high=1.)
        init_x = self.expected_vs * t
        init_delta_y = self.np_random.normal(0, 1)
        init_y = self.vehicle_dynamics.path.compute_path_y(t) + init_delta_y
        init_delta_phi = self.np_random.normal(0, np.pi / 9)
        init_phi = self.vehicle_dynamics.path.compute_path_phi(t) + init_delta_phi
        beta = self.np_random.normal(0, 0.15)
        init_v_y = self.expected_vs * np.tan(beta)
        init_r = self.np_random.normal(0, 0.3)
        obs = np.array([init_v_y, init_r, init_delta_y, init_delta_phi], dtype=np.float32)

        for i in range(self.prediction_horizon - 1):
            ref_y = self.vehicle_dynamics.path.compute_path_y(t + (i + 1)/self.base_frequency)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(t + (i + 1)/self.base_frequency)
            ref_obs = np.array([init_y - ref_y, init_phi - ref_phi], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))
        self.obs = obs
        self.state = np.array([init_v_y, init_r, init_y, init_phi, t], dtype=np.float32)
        return self.obs

    def step(self, action: np.ndarray, adv_action=None):  # think of action is in range [-1, 1]
        steer_norm = action
        action = steer_norm * 1.2 * np.pi / 9
        reward = self.vehicle_dynamics.compute_rewards(self.obs, action)
        self.state, self.obs = self.vehicle_dynamics.simulation(self.state, action,
                                             base_freq=self.base_frequency)
        self.done = self.judge_done(self.state)

        state = np.array(self.state, dtype=np.float32)
        return self.obs, reward, self.done, {"state":state}

    def judge_done(self, state):
        v_ys, rs, ys, phis, t = state[0], state[1], state[2], \
                                                   state[3], state[4]

        done = (np.abs(ys - self.vehicle_dynamics.path.compute_path_y(t)) > 3) | \
               (np.abs(phis - self.vehicle_dynamics.path.compute_path_phi(t)) > np.pi / 4.)
        return done

    def close(self):
        pass

    def render(self, mode='human'):
        pass


def env_creator(**kwargs):
    """
    make env `pyth_veh2dofconti`
    """
    return TimeLimit(SimuVeh2dofconti(**kwargs), 100)

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
