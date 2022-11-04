import gym
import numpy as np

class ConvertType(gym.Wrapper):
    """
        This wrapper converts the data type of the action and observation to satisfy the requirements of the  original
        environment and the gops interface.
    """
    def __init__(self, env):
        super(ConvertType, self).__init__(env)
        self.obs_data_tpye = env.observation_space.dtype
        self.act_data_type = env.action_space.dtype
        self.gops_data_type = np.float32

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs.astype(self.gops_data_type)
        return obs, info

    def step(self, action):
        action = action.astype(self.act_data_type)
        obs, rew, done, info = super(ConvertType, self).step(action)
        obs = obs.astype(self.gops_data_type)
        return obs, rew, done, info

