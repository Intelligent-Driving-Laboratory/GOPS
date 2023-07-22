#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  An adapter that converts a gym.Env into a gymnasium.Env
#  Update: 2023-07-22, Zhilong Zheng: create Gym2Gymnasium


import gym
from gym import spaces
from gym.core import Env
import gymnasium
from gymnasium.core import Env as GymnasiumEnv


class Gym2Gymnasium(gym.Wrapper, GymnasiumEnv):
    """
    An adapter that converts a gym.Env into a gymnasium.Env.
    """
    def __init__(self, env: Env):
        gym.Wrapper.__init__(self, env)

        # convert gym.spaces to gymnasium.spaces
        def convert_gym_space(space: spaces.Space) -> gymnasium.spaces.Space:
            if isinstance(space, spaces.Box):
                return gymnasium.spaces.Box(
                    low=space.low,
                    high=space.high,
                    dtype=space.dtype,
                    shape=space.shape,
                )
            elif isinstance(space, spaces.Discrete):
                return gymnasium.spaces.Discrete(n=space.n)
            elif isinstance(space, spaces.MultiBinary):
                return gymnasium.spaces.MultiBinary(n=space.n)
            elif isinstance(space, spaces.MultiDiscrete):
                return gymnasium.spaces.MultiDiscrete(nvec=space.nvec)
            elif isinstance(space, spaces.Tuple):
                return gymnasium.spaces.Tuple(
                    tuple([convert_gym_space(s) for s in space.spaces])
                )
            elif isinstance(space, spaces.Dict):
                return gymnasium.spaces.Dict(
                    {k: convert_gym_space(v) for k, v in space.spaces.items()}
                )
            else:
                raise NotImplementedError(
                    f"Unsupported gym space type: {type(space)}"
                )

        self.observation_space = convert_gym_space(self.env.observation_space)
        self.action_space = convert_gym_space(self.env.action_space)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        truncated = info.get("TimeLimit.truncated", False)
        info["TimeLimit.truncated"] = truncated
        terminated = done ^ truncated
        return observation, reward, terminated, truncated, info
    