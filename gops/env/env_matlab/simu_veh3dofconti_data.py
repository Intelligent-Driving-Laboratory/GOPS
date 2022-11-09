#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Simulink cartpole environment
#  Update Date: 2021-07-011, Wenxuan Wang: create simulink environment
from typing import Optional, List

from gym import spaces
import gym
from gym.utils import seeding

from gops.env.env_matlab.resources.simu_vehicle3dof_v2 import vehicle3dof
import numpy as np


class RefCurve:
    def __init__(self,
                 ref_a: List,
                 ref_t: List,
                 ref_fai: List,
                 ref_v,
                 ):
        self.A = ref_a
        self.T = ref_t
        self.fai = ref_fai
        self.V = ref_v

    def cal_reference(self, pos_x):
        pos_y = 0
        k_y = 0
        for items in zip(self.A, self.T, self.fai):
            pos_y += items[0] * np.sin(2 * np.pi / items[1] * pos_x + items[2])
            k_y += 2 * np.pi / items[1] * items[0] * np.cos(2 * np.pi / items[1] * pos_x + items[2])
        return pos_y, np.arctan(k_y), self.V


class SimuVeh3dofconti(gym.Env, ):
    def __init__(self, **kwargs):
        spec = vehicle3dof._env.EnvSpec(
            id='SimuVeh3dofConti-v0',
            max_episode_steps=kwargs["Max_step"],
            terminal_bonus_reward=kwargs["punish_done"],
            strict_reset=True
        )
        self.env = vehicle3dof.GymEnv(spec)
        self.dt = 0.01
        self.is_adversary = kwargs.get("is_adversary", False)
        self.obs_scale = np.array(kwargs["obs_scaling"])
        self.act_scale = np.array(kwargs["act_scaling"])
        self.act_max = np.array(kwargs["act_max"])
        self.done_range = kwargs['done_range']
        self.punish_done= kwargs["punish_done"]
        self.use_ref = kwargs['ref_info']
        self.ref_horizon = kwargs["ref_horizon"]
        self._state = None

        obs_low = self.obs_scale * np.array([-9999,-9999,-9999,-9999,-9999,-9999])
        ref_pos_low = -self.obs_scale[1] * self.done_range[0] * np.ones(self.ref_horizon)
        ref_phi_low = -self.obs_scale[4] * self.done_range[2] * np.ones(self.ref_horizon)
        if self.use_ref == "None":
            self.observation_space = spaces.Box(obs_low, -obs_low)
        elif self.use_ref == "Pos":
            obs_low = np.concatenate([obs_low, ref_pos_low])
            self.observation_space = spaces.Box(obs_low, -obs_low)
        elif self.use_ref == "Both":
            obs_low = np.concatenate([obs_low, ref_pos_low])
            obs_low = np.concatenate([obs_low, ref_phi_low])
            self.observation_space = spaces.Box(obs_low, -obs_low)
        else:
            raise NotImplementedError

        # Inherit or override with a user provided space
        self.action_space = self.env.action_space
        self.action_space = spaces.Box(
            -self.act_scale * self.act_max,
            self.act_scale * self.act_max,
        )
        # Split RNG, if randomness is needed
        self.rng = np.random.default_rng()

        self.reward_bias = kwargs["rew_bias"]
        self.reward_bound = kwargs["rew_bound"]
        self.act_repeat = kwargs["act_repeat"]
        self.rand_bias = kwargs["rand_bias"]
        self.rand_center = kwargs["rand_center"]
        ref_A = kwargs['ref_A']
        ref_T = kwargs['ref_T']
        ref_fai = kwargs['ref_fai']
        ref_V = kwargs['ref_V']

        self.Q = np.array(kwargs['punish_Q'])
        self.R = np.array(kwargs['punish_R'])
        self.ref_curve = RefCurve(ref_A, ref_T, ref_fai, ref_V)

        self.rand_low = np.array(self.rand_center) - np.array(self.rand_bias)
        self.rand_high = np.array(self.rand_center) + np.array(self.rand_bias)
        self.seed()
        self.reset()

    def reset(self):
        def callback():
            """Custom reset logic goes here."""
            # Modify your parameter
            # e.g. self.env.model_class.foo_InstP.your_parameter
            self._state = np.random.uniform(low=self.rand_low, high=self.rand_high)
            self.env.model_class.vehicle3dof_InstP.x_ini[:] = self._state
            self.env.model_class.vehicle3dof_InstP.ref_V = np.array(self.ref_curve.V)
            self.env.model_class.vehicle3dof_InstP.ref_fai[:] = np.array(self.ref_curve.fai)
            self.env.model_class.vehicle3dof_InstP.ref_A[:] = np.array(self.ref_curve.A)
            self.env.model_class.vehicle3dof_InstP.ref_T[:] = np.array(self.ref_curve.T)
            self.env.model_class.vehicle3dof_InstP.done_range[:] = np.array(self.done_range)
            self.env.model_class.vehicle3dof_InstP.punish_Q[:] = self.Q
            self.env.model_class.vehicle3dof_InstP.punish_R[:] = self.R

        # Reset takes an optional callback
        # This callback will be called after model & parameter initialization
        # and before taking first step.
        state = self.env.reset(callback)
        obs = self.postprocess(state)
        return obs

    def _physics_step(self, action: np.ndarray):
        state, reward, done, info = self.env.step(action)
        self._state = state
        return state, reward, done, info

    def step(self, action: np.ndarray):
        # Preprocess action here
        action_real = self.preprocess(action)
        # print("act is")
        # print(action_real)
        sum_reward = 0
        for idx in range(self.act_repeat):
            state, reward, done, info = self._physics_step(action_real)
            sum_reward += self.reward_shaping(reward)
            # print("reward is")
            # print(self.reward_shaping(reward))
            if done:
                sum_reward += self.punish_done
                #print("done")
                break
        # Postprocess (obs, reward, done, info) here
        obs = self.postprocess(state)
        return obs, sum_reward, done, info

    def preprocess(self, action):
        action_real = action / self.act_scale
        return action_real

    def postprocess(self, state):
        ref_y, ref_phi, ref_v = self.ref_curve.cal_reference(state[0])
        obs = np.zeros(self.observation_space.shape)
        obs[0] = state[0]
        obs[1] = state[1]-ref_y
        obs[2] = state[2] - ref_v
        obs[3] = state[3]
        obs[4] = state[4] - ref_phi
        obs[5] = state[5]
        obs[0:6] = obs[0:6]*self.obs_scale
        if self.use_ref =="Pos":
            x_pre = state[0] + ref_v * self.dt * self.act_repeat * np.linspace(1, self.ref_horizon, self.ref_horizon)
            y_pre, _, _ = self.ref_curve.cal_reference(x_pre)
            obs_y_pre = (state[1] - y_pre) * self.obs_scale[1]
            obs[6:] = obs_y_pre
        elif self.use_ref=="Both":
            x_pre = state[0] + ref_v * self.dt * self.act_repeat * np.linspace(1, self.ref_horizon, self.ref_horizon)
            y_pre, phi_pre, _ = self.ref_curve.cal_reference(x_pre)
            obs_y_pre = (state[1] - y_pre) * self.obs_scale[1]
            obs_phi_pre = (state[4] - phi_pre) * self.obs_scale[4]
            obs[6:] = np.concatenate([obs_y_pre, obs_phi_pre])
        return obs

    def reward_shaping(self, origin_reward):
        # print("reward is")
        # print(origin_reward)
        modified_reward = origin_reward
        if modified_reward <= -self.reward_bound:
            modified_reward = -self.reward_bound
        modified_reward = modified_reward + self.reward_bias
        return modified_reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    import gym
    import numpy as np

    env_config = {"ref_A": [0.0, 0.0, 0],
                  "ref_T": [100., 200., 400.],
                  "ref_V": 0.,
                  "ref_fai": [0, np.pi / 6, np.pi / 3],
                  "ref_horizon": 20,
                  "ref_info": "None",
                  "Max_step": 1000,
                  "act_repeat": 1,
                  "obs_scaling": [1, 1, 1, 1, 1, 1],
                  "act_scaling": [1, 1, 1],
                  "act_max": [20 * np.pi / 180, 3000, 3000],
                  "rew_bias": 5.,
                  "rew_bound": 5.,
                  "rand_center": [0, 0, 20, 0, 0, 0],
                  "rand_bias": [0, 0, 0, 0, 0, 0],
                  "done_range": [1000, 200, 100],
                  "punish_done": 0.,
                  "punish_Q": [1, 1, 10, 0.5],
                  "punish_R": [5, 1e-6, 1e-6],
                  }
    env = SimuVeh3dofconti(**env_config)
    s = env.reset()
    print(s)
    print(env.env.model_class.vehicle3dof_InstP.done_range)
    for i in range(1000):
        print(i)
        a = np.array([0.02, 1000, 1000])
        sp, r, d, _ = env.step(a)
        print(sp)
        s = sp
