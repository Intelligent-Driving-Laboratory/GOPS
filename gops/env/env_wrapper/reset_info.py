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
