#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment


import gym
import numpy as np

from gops.env.env_ocp.pyth_base_data import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData


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
                                   )
        l_f, l_r, mass, g = self.vehicle_params['l_f'], self.vehicle_params['l_r'], \
                        self.vehicle_params['m'], self.vehicle_params['g']
        F_zf, F_zr = l_r * mass * g / (l_f + l_r), l_f * mass * g / (l_f + l_r)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.expected_vs = 10.
        self.pre_horizon = kwargs['pre_horizon']

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
        next_state = [x + delta_t * (u * np.cos(phi) - v * np.sin(phi)),
                      y + delta_t * (u * np.sin(phi) + v * np.cos(phi)),
                      phi + delta_t * w,
                      u + delta_t * a_x,
                      (m * v * u + delta_t * (
                                  l_f * k_f - l_r * k_r) * w - delta_t * k_f * steer * u - delta_t * m * np.square(
                          u) * w) / (m * u - delta_t * (k_f + k_r)),
                      (I_z * w * u + delta_t * (l_f * k_f - l_r * k_r) * v - delta_t * l_f * k_f * steer * u) / (
                                  I_z * u - delta_t * (np.square(l_f) * k_f + np.square(l_r) * k_r))
                      ]
        return next_state

    def prediction(self, x_1, u_1, frequency):
        x_next = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next

    def simulation(self, states, actions, base_freq, ref_traj: MultiRefTrajData, t: float, path_num: int, u_num: int):
        state_next = self.prediction(states, actions, base_freq)
        x, y, phi, u, v, w = state_next[0], state_next[1], state_next[2], state_next[3], state_next[4], state_next[5]
        path_x, path_y, path_phi = ref_traj.compute_x(t, path_num, u_num), \
                                   ref_traj.compute_y(t, path_num, u_num), \
                                   ref_traj.compute_phi(t, path_num, u_num)
        path_u = ref_traj.compute_u(t, path_num, u_num)
        obs = np.array([x - path_x, y - path_y, phi - path_phi, u - path_u, v, w], dtype=np.float32)
        for i in range(self.pre_horizon):
            ref_x = ref_traj.compute_x(t + (i + 1) / base_freq, path_num, u_num)
            ref_y = ref_traj.compute_y(t + (i + 1) / base_freq, path_num, u_num)
            ref_obs = np.array([x - ref_x, y - ref_y], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))

        if state_next[2] > np.pi:
            state_next[2] -= 2 * np.pi
        if state_next[2] <= -np.pi:
            state_next[2] += 2 * np.pi

        return state_next, obs

    def compute_rewards(self, obs, actions):  # obses and actions are tensors
        delta_x, delta_y, delta_phi, delta_u, v, w = obs[0], obs[1], obs[2], \
                                                   obs[3], obs[4], obs[5]
        steers, a_xs = actions[0], actions[1]
        devi_y = -np.square(delta_y)
        devi_phi = -np.square(delta_phi)
        punish_yaw_rate = -np.square(w)
        punish_steer = -np.square(steers)
        punish_a_x = -np.square(a_xs)
        punish_x = -np.square(delta_x)
        punish_u = -np.square(delta_u)
        rewards = 0.1 * devi_y + 0.01 * punish_u + 0.01 * devi_phi + 0.01 * punish_yaw_rate + \
                  0.01 * punish_steer + 0.01 * punish_a_x + 0.04 * punish_x

        return rewards


class SimuVeh3dofconti(PythBaseEnv):
    def __init__(self, path_para:dict = None,
                 u_para:dict = None, **kwargs):
        self.vehicle_dynamics = VehicleDynamics(**kwargs)

        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_x, delta_y, delta_phi, delta_u, v, w]
            init_high = np.array([2, 1, np.pi / 3, 5, 5 * 0.25, 0.9], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(SimuVeh3dofconti, self).__init__(work_space=work_space, **kwargs)

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)
        self.pre_horizon = kwargs["pre_horizon"]
        self.base_frequency = 10
        self.state_dim = 6
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (2 * self.pre_horizon + self.state_dim)),
            high=np.array([np.inf] * (2 * self.pre_horizon + self.state_dim)),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-np.pi / 6, -3]),
                                           high=np.array([np.pi / 6, 3]),
                                           dtype=np.float32)
        self.max_episode_steps = 200
        self.obs = None
        self.state = None
        self.path_num = None
        self.para = None
        self.u_num = None
        self.u_para = None
        self.t = None
        self.info_dict = {
            "state": {"shape": self.state_dim, "dtype": np.float32},
            "ref": {"shape": (2,), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
        }
        self.seed()
        self.ref_traj = MultiRefTrajData(path_para, u_para)

    @property
    def additional_info(self):
        return self.info_dict

    def reset(self, init_state=None, ref_time=None, path_num=None, u_num=None, **kwargs):
        init_y = None
        init_phi = None
        init_u = None
        init_v = None
        init_w = None
        init_x = None
        obs = None
        if (init_state is None) & (ref_time is None) & (path_num is None) & (u_num is None):
            obs = self.sample_initial_state()
            delta_x, delta_y, delta_phi, delta_u, v, w = obs
            flag = [0, 1, 2, 3]
            self.path_num = self.np_random.choice(flag)
            u_flag = [0, 1]
            self.u_num = self.np_random.choice(u_flag)
            ref_time = 20. * self.np_random.uniform(low=0., high=1.)
            self.t = ref_time
            u = self.ref_traj.compute_u(self.t, self.path_num, self.u_num)
            path_x = self.ref_traj.compute_x(self.t, self.path_num, self.u_num)
            init_x = path_x + delta_x
            init_y = self.ref_traj.compute_y(self.t, self.path_num, self.u_num) + delta_y
            init_phi = self.ref_traj.compute_phi(self.t, self.path_num, self.u_num) + delta_phi
            init_u = u + delta_u
            init_v = v
            init_w = w
        elif (init_state is not None) & (ref_time is not None) & (path_num is not None) & (u_num is not None):
            self.path_num = path_num
            self.u_num = u_num
            self.t = ref_time
            init_x, init_y, init_phi, init_u, init_v, init_w = init_state[0], init_state[1], init_state[2], init_state[3], init_state[4], init_state[5]
            delta_x = init_x - self.ref_traj.compute_x(self.t, self.path_num, self.u_num)
            delta_y = init_y - self.ref_traj.compute_y(self.t, self.path_num, self.u_num)
            delta_phi = init_phi - self.ref_traj.compute_phi(self.t, self.path_num, self.u_num)
            delta_u = init_u - self.ref_traj.compute_u(self.t, self.path_num, self.u_num)
            obs = np.array([delta_x, delta_y, delta_phi, delta_u, init_v, init_w], dtype=np.float32)
        else:
            print("reset error")

        for i in range(self.pre_horizon):
            ref_x = self.ref_traj.compute_x(self.t + (i + 1) / self.base_frequency, self.path_num, self.u_num)
            ref_y = self.ref_traj.compute_y(self.t + (i + 1) / self.base_frequency, self.path_num, self.u_num)
            ref_obs = np.array([init_x - ref_x, init_y - ref_y], dtype=np.float32)
            obs = np.hstack((obs, ref_obs))
        self.obs = obs
        self.state = np.array([init_x, init_y, init_phi, init_u, init_v, init_w], dtype=np.float32)
        return self.obs, self.info

    def step(self, action: np.ndarray, adv_action=None):  # think of action is in range [-1, 1]
        steer_norm, a_x_norm = action[0], action[1]
        action = np.stack([steer_norm, a_x_norm], 0)
        reward = self.vehicle_dynamics.compute_rewards(self.obs, action)
        self.t = self.t + 1.0 / self.base_frequency
        self.state, self.obs = self.vehicle_dynamics.simulation(
            self.state, action, self.base_frequency, self.ref_traj, self.t, self.path_num, self.u_num)
        self.done = self.judge_done(self.state, self.t)
        if self.done:
            reward = reward - 100

        return self.obs, reward, self.done, self.info

    def judge_done(self, veh_state, t):
        x, y, phi = veh_state[0], veh_state[1], veh_state[2]
        done = (np.abs(y - self.ref_traj.compute_y(t, self.path_num, self.u_num)) > 2) |\
               (np.abs(x - self.ref_traj.compute_x(t, self.path_num, self.u_num)) > 5) |\
               (np.abs(phi - self.ref_traj.compute_phi(t, self.path_num, self.u_num)) > np.pi)
        return done

    @property
    def info(self):
        state = np.array(self.state, dtype=np.float32)
        x_ref = self.ref_traj.compute_x(self.t, self.path_num, self.u_num)
        y_ref = self.ref_traj.compute_y(self.t, self.path_num, self.u_num)
        ref = np.array([x_ref, y_ref], dtype=np.float32)
        return {
            "state": state,
            "ref": ref,
            "path_num": self.path_num,
            "u_num": self.u_num,
            "ref_time": self.t,
        }


def env_creator(**kwargs):
    """
    make env `pyth_veh3dofconti`
    """
    return SimuVeh3dofconti(path_para=None,
                 u_para=None, **kwargs)
