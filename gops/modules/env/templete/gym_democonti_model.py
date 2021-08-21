#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Sun Hao
#  Description: Acrobat Environment
#
#  Update Date: 2021-05-55, Sun Hao: create environment


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
