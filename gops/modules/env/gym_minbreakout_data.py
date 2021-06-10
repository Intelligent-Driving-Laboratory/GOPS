from gym.spaces import Discrete, Box
from gym import Env
from gym.wrappers.time_limit import TimeLimit
import numpy as np


class _Minbreakout(Env):
    def __init__(self):
        try:
            from minatar import Environment
        except:
            raise NotImplementedError('MinAtar is not installed')
        self.env = Environment('breakout')
        num_act = self.env.num_actions()
        obs_shape = self.env.state_shape()

        self.observation_space = Box(0, 1, obs_shape)
        self.action_space = Discrete(num_act)

    def reset(self):
        self.env.reset()
        s = self.env.state()
        return np.array(s, dtype=float)

    def step(self, a):
        reward, terminal = self.env.act(a)
        s = self.env.state()
        return np.array(s, dtype=float), reward, terminal, None

    def render(self, *args, **kwargs):
        self.env.display_state(20)


def env_creator():
    e = _Minbreakout()
    return TimeLimit(e, 100)
