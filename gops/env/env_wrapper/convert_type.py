#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data type environment wrapper for converting data type
#  Update: 2022-10-26, Wenxuan Wang: create convert type wrapper


import gym
import numpy as np
from gym.core import ObsType, ActType
from typing import Tuple


class ConvertType(gym.Wrapper):
    """Wrapper converts data type of action and observation to satisfy requirements of original
        environment and gops interface.
    :param env: data type environment.
    """

    def __init__(self, env):
        super(ConvertType, self).__init__(env)
        self.obs_data_tpye = env.observation_space.dtype
        self.act_data_type = env.action_space.dtype
        self.gops_data_type = np.float32

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset(**kwargs)
        obs = obs.astype(self.gops_data_type)
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        action = action.astype(self.act_data_type)
        obs, rew, done, info = super(ConvertType, self).step(action)
        obs = obs.astype(self.gops_data_type)
        return obs, rew, done, info
