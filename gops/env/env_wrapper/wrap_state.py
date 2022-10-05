import gym
import numpy as np

class StateData(gym.Wrapper):
    """
        This wrapper ensures that the environment has a "state" property.
        If the original environment does not have one, the current observation is returned when calling state.
    """
    def __init__(self, env):
        super(StateData, self).__init__(env)
        self.current_obs = None

    def reset(self, **kwargs):
        obs = super(StateData, self).reset(**kwargs)
        self.current_obs = obs
        return obs

    def step(self, action):
        obs, rew, done, info = super(StateData, self).step(action)
        self.current_obs = obs
        return obs, rew, done, info

    @property
    def state(self):
        if hasattr(self.env, "state"):
            return np.array(self.env.state,dtype=np.float32)
        else:
            return self.current_obs
