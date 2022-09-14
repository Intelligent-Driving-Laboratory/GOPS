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
import numpy as np
import torch



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
    """
    make env `pyth_veh2dofconti`
    """
    return SimuVeh2dofconti(**kwargs)


def scale_obs(obs):
    obs_scale = [1., 2., 1., 2.4]
    v_ys, rs, delta_ys, delta_phis = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    lists_to_stack = [v_ys * obs_scale[0], rs * obs_scale[1],
                      delta_ys * obs_scale[2], delta_phis * obs_scale[3]]
    return torch.stack(lists_to_stack, 1)
