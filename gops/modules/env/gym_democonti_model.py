#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao

# env.py
#
#
#
#


import gym
from gym.utils import seeding
import numpy as np
import torch


class GymDemocontiModel(gym.Env):
    def __init__(self):
        # define all the parameters here
        self.state_dim = None
        self.action_dim = None
        self.lb = None
        self.hb = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action:torch.Tensor):
        self.state = self.forward(state=self.state,action= action)
        cost = self.cost(state=self.state.numpy(),action=action)
        return self.state, cost

    def reset(self,state=None)->torch.Tensor:
        if state is None:
            state = self.np_random.uniform(low=self.lb, high=self.hb, size=(self.state_dim,))
            self.state = torch.from_numpy(state)
        else:
            self.state = torch.from_numpy(state)
        self.state.requires_grad = True
        return self.state


    def forward(self,state:torch.Tensor,action:torch.Tensor)->torch.Tensor:
        state_next = None
        #  x_next = f(x,u)
        #  define the forward function here
        assert isinstance(state_next,torch.Tensor)
        return state_next

    def cost(self,state:np.ndarray,action:torch.Tensor)->torch.Tensor:
        cost = None
        # cost = g(x,y)
        # define the cost function here
        assert isinstance(cost, torch.Tensor)
        return cost


if __name__ == "__main__":
    env = GymDemocontiModel()
