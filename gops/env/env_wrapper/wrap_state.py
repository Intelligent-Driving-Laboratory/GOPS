#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data type environment wrapper
#  Update: 2022-10-27, Yujie Yang: create state data wrapper


import gym
import numpy as np

from typing import Tuple
from gym.core import ObsType, ActType


class StateData(gym.Wrapper):
    """
    Wrapper ensures that environment has "state" property.
    If original environment does not have one, current observation is returned when calling state.
    """

    def __init__(self, env):
        super(StateData, self).__init__(env)
        self.current_obs = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = super(StateData, self).step(action)
        self.current_obs = obs
        return obs, rew, done, info

    @property
    def state(self):
        if hasattr(self.env, "state"):
            return np.array(self.env.state, dtype=np.float32)
        else:
            return self.current_obs
