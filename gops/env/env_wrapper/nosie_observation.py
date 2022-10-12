from typing import TypeVar, Tuple

import gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class NoiseData(gym.Wrapper):
    """
        noise_type has two type: "normal" and "uniform"
        if noise_type == "normal", noise_data means Mean and Standard deviation of Normal distribution
        if noise_type == "uniform", noise_data means Upper and Lower bounds of Uniform distribution
    """
    def __init__(self, env, noise_type, noise_data):
        super(NoiseData, self).__init__(env)
        assert noise_type in ["normal", "uniform"]
        assert len(noise_data) == 2 and len(noise_data[0]) == env.observation_space.shape[0]
        self.noise_type = noise_type
        self.noise_data = np.array(noise_data, dtype=np.float32)

    def observation(self, observation):
        if self.noise_type is None:
            return observation
        elif self.noise_type == "normal":
            return observation + np.random.normal(loc=self.noise_data[0], scale=self.noise_data[1])
        elif self.noise_type == "uniform":
            return observation + np.random.uniform(low=self.noise_data[0], high=self.noise_data[1])

    def reset(self, **kwargs):
        obs = super(NoiseData, self).reset(**kwargs)
        obs_noised = self.observation(obs)
        return obs_noised

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        obs_noised = self.observation(obs)
        return obs_noised, r, d, info
