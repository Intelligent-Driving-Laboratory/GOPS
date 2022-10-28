#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Aircraft Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

from gops.env.env_matlab.resources.simu_lqs2a1.lqs2a1 import GymEnv
from gops.env.env_matlab.resources.simu_lqs2a1.lqs2a1._env import EnvSpec
from gops.env.env_matlab.resources.simu_lqs2a1 import lqs2a1

from gym import spaces
import gym
from gym.utils import seeding
import numpy as np

class Lqs2a1(gym.Env):
    def __init__(self, **kwargs):
        spec = EnvSpec(
            id="SimuLqs2a1Conti-v0",
            max_episode_steps=kwargs["Max_step"],
            terminal_bonus_reward=kwargs["punish_done"],
            strict_reset=True
        )
        self.env = GymEnv(spec)

        # Inherit or override with a user provided space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Split RNG, if randomness is needed
        self.rng = np.random.default_rng()
        self.Q = np.array(kwargs['punish_Q'])
        self.R = np.array(kwargs['punish_R'])
        self.rand_bias = kwargs["rand_bias"]
        self.rand_center = kwargs["rand_center"]
        self.rand_low = np.array(self.rand_center) - np.array(self.rand_bias)
        self.rand_high = np.array(self.rand_center) + np.array(self.rand_bias)
        self.seed()
        self.reset()
    
    def reset(self):
        def callback():
            """Custom reset logic goes here."""
            # Modify your parameter
            # e.g. self.env.model_class.foo_InstP.your_parameter
            self._state = np.random.uniform(low=self.rand_low, high=self.rand_high) ##todo
            self.env.model_class.lqs2a1_InstP.x_ini[:] = self._state
            self.env.model_class.lqs2a1_InstP.Q[:] = self.Q
            self.env.model_class.lqs2a1_InstP.R = self.R

        # Reset takes an optional callback
        # This callback will be called after model & parameter initialization
        # and before taking first step.
        return self.env.reset(callback)

    def step(self, action):
        # Preprocess action here
        obs, done, reward, info = self.env.step(action)
        # Postprocess (obs, reward, done, info) here
        return obs, done, reward, info

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return self.env.seed(seed) if seed is not None else self.env.seed()

def env_creator(**kwargs):
    return Lqs2a1(**kwargs)


if __name__ == "__main__":
    import gym
    import numpy as np
    import time
    env_config = {
                  "Max_step": 200,
                  "rand_center": [0, 0],
                  "rand_bias": [1, 1],
                  "punish_Q": [2, 1],
                  "punish_R": [1],
                  "punish_done": 0.,
                  }
    env = Lqs2a1(**env_config)
    s = env.reset()
    print(s,'sssssssssssssss')
    # print(env.env.model_class.dof14model_InstP.done_range)
    for i in range(1000):
        # print(i)
        a = np.array([0.02])
        sp, d, re, _ = env.step(a)
        print(i)
        print(sp)
        s = sp

