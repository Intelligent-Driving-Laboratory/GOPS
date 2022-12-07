#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Simulink LQs2a1 Environment
#  Update: 2022-10-27, Genjin Xie: create environment
#  Update: 2022-11-03, Xujie Song: modify __init__ and reset

from typing import Any, Optional, Tuple, Sequence

from gops.env.env_matlab.resources.simu_lqs2a1.lqs2a1 import GymEnv
from gops.env.env_matlab.resources.simu_lqs2a1.lqs2a1._env import EnvSpec
from gops.env.env_ocp.pyth_base_data import PythBaseEnv
import numpy as np


class Lqs2a1(PythBaseEnv):
    def __init__(self, **kwargs: Any):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            init_high = np.array([1, 1], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(Lqs2a1, self).__init__(work_space=work_space, **kwargs)

        spec = EnvSpec(
            id="SimuLqs2a1Conti-v0",
            terminal_bonus_reward=kwargs["punish_done"],
            strict_reset=True,
        )
        self.env = GymEnv(spec)
        self.max_episode_steps = kwargs["max_episode_steps"]

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.rng = np.random.default_rng()
        self.Q = np.array(kwargs["punish_Q"])
        self.R = np.array(kwargs["punish_R"])
        self.rand_bias = kwargs["rand_bias"]
        self.rand_center = kwargs["rand_center"]
        self.rand_low = np.array(self.rand_center) - np.array(self.rand_bias)
        self.rand_high = np.array(self.rand_center) + np.array(self.rand_bias)
        self.seed()
        self.reset()

    def reset(
        self, *, init_state: Optional[Sequence] = None, **kwargs: Any
    ) -> Tuple[np.ndarray]:
        def callback():
            """Custom reset logic goes here."""
            # Modify  parameters
            # e.g. self.env.model_class.foo_InstP.your_parameter
            if init_state is None:
                self._state = self.sample_initial_state()
            else:
                self._state = init_state
            self.env.model_class.lqs2a1_InstP.x_ini[:] = self._state
            self.env.model_class.lqs2a1_InstP.Q[:] = self.Q
            self.env.model_class.lqs2a1_InstP.R = self.R

        # Reset takes an optional callback
        # This callback will be called after model & parameter initialization and before taking first step.
        return self.env.reset(callback)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, bool, float, dict]:
        obs, done, reward, info = self.env.step(action)
        return obs, done, reward, info

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return self.env.seed(seed) if seed is not None else self.env.seed()


def env_creator(**kwargs):
    return Lqs2a1(**kwargs)
