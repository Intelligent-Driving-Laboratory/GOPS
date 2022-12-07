#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data type environment wrapper
#  Update: 2022-10-27, Yujie Yang: create reset info wrapper


from typing import Tuple

import gym
from gym.core import ObsType


class ResetInfoData(gym.Wrapper):
    """
    This wrapper ensures that the 'reset' method returns a tuple (obs, info).
    """

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            return ret
        else:
            return ret, {}
