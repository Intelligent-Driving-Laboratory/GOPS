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
        self.path = ReferencePath()
        self.prediction_horizon = 10

    def f_xu(self, states, actions, tau):
        v_x, v_y, r, delta_y, delta_phi, x = states[0], states[1], states[2], \
                                             states[3], states[4], states[5]
        steer, a_x = actions[0], actions[1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x))
        F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
        miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr
        alpha_f = np.arctan((v_y + a * r) / v_x) - steer
        alpha_r = np.arctan((v_y - b * r) / v_x)
        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                                  a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * np.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                                  tau * (np.square(a) * C_f + np.square(b) * C_r) - I_z * v_x),
                      delta_y + tau * (v_x * np.sin(delta_phi) + v_y * np.cos(delta_phi)),
                      delta_phi + tau * r,
                      x + tau * (v_x * np.cos(delta_phi) - v_y * np.sin(delta_phi)),
                      ]
        alpha_f_bounds, alpha_r_bounds = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bounds = miu_r * g / np.abs(v_x)
        other = [alpha_f, alpha_r, next_state[2], alpha_f_bounds, alpha_r_bounds, r_bounds]
        return next_state, other


    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params

    def simulation(self, states, actions, base_freq):
        state_next, others = self.prediction(states, actions, base_freq)
        states[0] = np.clip(states[0], 1, 35)
        v_x, v_y, r, y, phi, x, t = states[0], states[1], states[2], states[3], states[4], states[5], states[6]
        path_x, path_y, path_phi = self.path.compute_path_x(t), \
                                   self.path.compute_path_y(t), \
                           self.path.compute_path_phi(t)
        obs = np.array([v_x - self.expected_vs, v_y, r, y - path_y, phi - path_phi, x - path_x], dtype=np.float32)
        for i in range(self.prediction_horizon - 1):
            ref_x = self.path.compute_path_x(t + (i + 1) / base_freq)
            ref_y = self.path.compute_path_y(t + (i + 1) / base_freq)
            ref_phi = self.path.compute_path_phi(t + (i + 1) / base_freq)
            ref_obs = np.array([x - ref_x, y - ref_y, phi - ref_phi], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))
        if state_next[-2] > self.path.period:
            state_next[-2] -= self.path.period
        if state_next[-2] < self.path.period:
            state_next[-2] += self.path.period
        if state_next[4] > np.pi:
            state_next[4] -= 2 * np.pi
        if state_next[4] <= -np.pi:
            state_next[4] += 2 * np.pi

        return state_next, obs, others

    def compute_rewards(self, obs, actions):  # obses and actions are tensors

        v_xs, v_ys, rs, delta_ys, delta_phis, xs = obs[0], obs[1], obs[2], \
                                                   obs[3], obs[4], obs[5]
        steers, a_xs = actions[0], actions[1]
        devi_v = -np.square(v_xs - self.expected_vs)
        devi_y = -np.square(delta_ys)
        devi_phi = -np.square(delta_phis)
        punish_yaw_rate = -np.square(rs)
        punish_steer = -np.square(steers)
        punish_a_x = -np.square(a_xs)

        rewards = 0.1 * devi_v + 0.4 * devi_y + 1 * devi_phi + 0.2 * punish_yaw_rate + \
                  0.5 * punish_steer + 0.5 * punish_a_x

        return rewards


class ReferencePath(object):
    def __init__(self):
        self.expect_v = 10
        self.period = 1200

    def compute_path_x(self, t):
        x = self.expect_v * t
        return x

    def compute_path_y(self, t):
        y = np.sin((1 / 30) * self.expect_v * t)
        return y

    def compute_path_phi(self, t):
        phi = (np.sin((1 / 30) * self.expect_v * (t + 0.001)) - np.sin((1 / 30) * self.expect_v * t)) / (
                    self.expect_v * 0.001)
        return np.arctan(phi)


class SimuVeh3dofconti(gym.Env,):
    def __init__(self, num_future_data=0, num_agent=1, **kwargs):

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)
        self.prediction_horizon = 10
        self.vehicle_dynamics = VehicleDynamics()
        self.num_agent = num_agent
        self.expected_vs = 10.
        self.base_frequency = 10
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (33)),
            high=np.array([np.inf] * (33)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.2 * np.pi / 9, -3]),
                                           high=np.array([1.2 * np.pi / 9, 3]),
                                           dtype=np.float32)
        self.obs = None
        self.state = None
        self.state_dim = 7
        self.info_dict = {"state": {"shape": self.state_dim, "dtype": np.float32}}
        self.seed()


    @property
    def additional_info(self):
        return self.info_dict

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        t = 60. * self.np_random.uniform(low=0., high=1.)
        path_x = self.vehicle_dynamics.path.compute_path_x(t)
        init_delta_x = self.np_random.normal(0, 2)
        init_x = path_x + init_delta_x
        init_delta_y = self.np_random.normal(0, 1)
        init_y = self.vehicle_dynamics.path.compute_path_y(t) + init_delta_y
        init_delta_phi = self.np_random.normal(0, np.pi / 9)
        init_phi = self.vehicle_dynamics.path.compute_path_phi(t) + init_delta_phi
        beta = self.np_random.normal(0, 0.15)
        init_r = self.np_random.normal(0, 0.3)
        init_v_x = self.np_random.uniform(low=5., high=15.)
        init_v_y = init_v_x * np.tan(beta)
        obs = np.array([init_v_x - self.expected_vs, init_v_y, init_r, init_delta_y, init_delta_phi, init_delta_x], dtype=np.float32)

        for i in range(self.prediction_horizon - 1):
            ref_x = self.vehicle_dynamics.path.compute_path_x(t + (i + 1) / self.base_frequency)
            ref_y = self.vehicle_dynamics.path.compute_path_y(t + (i + 1) / self.base_frequency)
            ref_phi = self.vehicle_dynamics.path.compute_path_phi(t + (i + 1) / self.base_frequency)
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
                                                                base_freq=self.base_frequency)
        self.done = self.judge_done(self.state, stability_related)
        state = np.array(self.state, dtype=np.float32)

        return self.obs, reward, self.done, {"state":state}

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
    """
    make env `pyth_veh3dofconti`
    """
    return TimeLimit(SimuVeh3dofconti(**kwargs), 100)

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