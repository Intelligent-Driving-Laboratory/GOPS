#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description:
#  Update: 2021.03.05, Shengbo LI (example, can be deleted)


"""


"""

import numpy as np
import torch


class EpsilonScheduler():
    """Epsilon-greedy scheduler with epsilon schedule."""

    def __init__(self, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=2000):
        """Create an EpsilonScheduler.

        For fixed epsilon-greedy policy, passing EPS_START equal to EPS_END.

        Args:
            EPS_START (float, optional): Epsilon when starting training. Defaults to 0.9.
            EPS_END (float, optional): Epsilon when training infinity steps. Defaults to 0.05.
            EPS_DECAY (float, optional): Exponential coefficient, larger for a slower decay rate (similar to time constant, but for steps). Defaults to 200.
        """
        self.start = EPS_START
        self.end = EPS_END
        self.decay = EPS_DECAY

    def sample(self, action, action_num, steps):
        """Choose an action based on epsilon-greedy policy.

        Args:
            action (any): Predicted action, usually greedy.
            action_num (int): Num of discrete actions.
            steps (int): Global training steps.

        Returns:
            any: Action chosen by psilon-greedy policy.
        """
        thresh = self.end + (self.start - self.end) * np.exp(-steps / self.decay)
        if np.random.random() > thresh:
            return action
        else:
            return np.random.randint(action_num)


class EpsilonGreedy():
    def __init__(self, epsilon, action_num):
        self.epsilon = epsilon
        self.action_num = action_num

    def sample(self, action):
        if np.random.random() > self.epsilon:
            return action
        else:
            return np.random.randint(self.action_num)


class GaussNoise():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, action):
        return action + np.random.normal(self.mean, self.std)
