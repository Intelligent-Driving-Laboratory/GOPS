#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data type environment wrapper that add noise to observation
#  Update: 2022-10-27, Congsheng Zhang: create noise observation wrapper


from typing import Tuple

import gym
import numpy as np
from gym.utils import seeding
from gym.core import ObsType, ActType


class NoiseData(gym.Wrapper):
    """Data type environment wrapper that add noise to observation.

    :param env: data type environment.
    :param str noise_type: distribution of noise, support Normal distribution and Uniform distribution.
    :param np.ndarray noise_data: if noise_type == "normal", noise_data means Mean and
        Standard deviation of Normal distribution. if noise_type == "uniform", noise_data means Upper
        and Lower bounds of Uniform distribution.
    """

    def __init__(self, env, noise_type: str, noise_data: list):
        super(NoiseData, self).__init__(env)
        assert noise_type in ["normal", "uniform"]
        assert (
            len(noise_data) == 2
            and len(noise_data[0]) == env.observation_space.shape[0]
        )
        self.noise_type = noise_type
        self.noise_data = np.array(noise_data, dtype=np.float32)

    def observation(self, observation):
        if self.noise_type is None:
            return observation
        elif self.noise_type == "normal":
            return observation + self.np_random.normal(
                loc=self.noise_data[0], scale=self.noise_data[1]
            )
        elif self.noise_type == "uniform":
            return observation + self.np_random.uniform(
                low=self.noise_data[0], high=self.noise_data[1]
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_noised = self.observation(obs)
        return obs_noised, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        obs_noised = self.observation(obs)
        return obs_noised, r, d, info

    def seed(self, seed=None):
        np_random, _ = seeding.np_random(seed)
        noise_seed = int(np_random.randint(2 ** 31))
        self.np_random, noise_seed = seeding.np_random(noise_seed)
        seeds = self.env.seed(seed)
        return seeds + [noise_seed]
