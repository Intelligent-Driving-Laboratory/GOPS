#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: base class for pyth environments
#  Update: 2022-10-06, Yujie Yang: create base environment
#  Update: 2022-10-13, Wenxuan Wang: add has_optimal_controller and control_policy

from typing import Any, Optional, Sequence

import gym
import numpy as np
from gym.utils import seeding


class PythBaseEnv(gym.Env):
    def __init__(
        self,
        work_space: Sequence,
        train_space: Optional[Sequence] = None,
        initial_distribution: str = "uniform",
        **kwargs: Any,
    ):
        self.work_space = np.array(work_space, dtype=np.float32)
        assert self.work_space.ndim == 2 and self.work_space.shape[0] == 2

        if train_space is not None:
            self.train_space = np.array(train_space, dtype=np.float32)
            assert self.train_space.ndim == 2 and self.train_space.shape[0] == 2
        else:
            self.train_space = self.work_space

        self.initial_distribution = initial_distribution

        self.mode = "train"

    def set_mode(self, mode):
        assert mode in ["train", "test"]
        self.mode = mode

    @property
    def has_optimal_controller(self):
        return False

    def control_policy(self, state, info):
        return NotImplementedError

    @property
    def init_space(self):
        if self.mode == "train":
            return self.train_space
        elif self.mode == "test":
            return self.work_space
        else:
            raise ValueError(f"Invalid mode: {self.mode}!")

    def sample_initial_state(self):
        if self.initial_distribution == "uniform":
            state = self.np_random.uniform(
                low=self.init_space[0], high=self.init_space[1]
            )
        elif self.initial_distribution == "normal":
            mean = (self.init_space[0] + self.init_space[1]) / 2
            std = (self.init_space[1] - self.init_space[0]) / 6
            state = self.np_random.normal(loc=mean, scale=std)
        else:
            raise ValueError(
                f"Invalid initial distribution: {self.initial_distribution}!"
            )
        return state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
