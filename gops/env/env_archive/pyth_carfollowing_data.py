#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Carfollowing Environment
#  Update Date: 2021-11-22, Yuhang Zhang: create environment


import math
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding
from gops.env.resources.car_following.car_following import CarFollowingDynamics

gym.logger.setLevel(gym.logger.ERROR)


class PythCarfollowingData:
    def __init__(self, **kwargs):

        self.dyn = CarFollowingDynamics()
        self.mode = "training"
        self.constraint_dim = 1

        lb_state = np.array([-np.inf, -np.inf, -np.inf])
        hb_state = -lb_state
        lb_action = np.array(
            [
                -4.0,
            ]
        )
        hb_action = np.array(
            [
                3.0,
            ]
        )

        self.action_space = spaces.Box(low=lb_action, high=hb_action)
        self.observation_space = spaces.Box(lb_state, hb_state)

        self.seed()
        self.state = self.reset()

        self.steps = 0

    def set_mode(self, mode):
        self.mode = mode

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        state_next = self.dyn.prediction(self.state, action)
        self.state = state_next

        reward = self.dyn.compute_reward(state_next, action)
        ############################################################################################
        # define the constraint here

        ################################################################################################################
        # define the ending condition here the format is just like isdone = l(next_state)

        isdone = state_next[2] < 2

        constraint = self.dyn.compute_cost(
            state_next, action
        )  # TODO: next state or state
        self.steps += 1
        info = {"TimeLimit.truncated": self.steps > 170, "constraint": constraint}
        return self.state, reward, isdone, info

    def reset(self):
        self.steps = 0
        if self.mode == "training":
            v_e = np.clip(self.np_random.uniform(0, 7), 0, 7)
            v_t = np.clip(self.np_random.uniform(2, 7), 2, 7)
            gap = np.clip(self.np_random.uniform(8, 15), 1, 20)

        elif self.mode == "testing":
            v_e = np.clip(self.np_random.uniform(0, 7), 0, 7)
            v_t = np.clip(self.np_random.uniform(2, 7), 2, 7)
            gap = np.clip(self.np_random.uniform(4, 15), 0, 20)
        else:
            raise ValueError("Wrong mode, please set mode to [training] or [testing]")
        self.state = np.array([v_e, v_t, gap])
        return self.state

    def render(self, n_window=1):
        pass

    def close(self):
        pass


def env_creator(**kwargs):
    return PythCarfollowingData()


if __name__ == "__main__":
    from pprint import pprint

    env = PythCarfollowingData()
    for i in range(100):
        a = env.action_space.sample()
        x, r, d, info = env.step(a)
        pprint([x, r, d, info])

        env.render()
