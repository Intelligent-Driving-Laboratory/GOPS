#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Wenxuan Wang
#  Description: Acrobat Environment
#

import math
import warnings
import torch
import numpy as np
from gym import spaces
from gym.utils import seeding


class GymDemocontiModel:
    def __init__(self):
        """
        you need to define parameters here
        """
        # define your custom parameters here

        # define common parameters here
        self.state_dim = None
        self.action_dim = None
        self.lb_state = None
        self.hb_state = None
        self.lb_action = None
        self.hb_action = None
        self.lb_init_state = None
        self.hb_init_state = None
        self.tau = None    # seconds between state updates

        # do not change the following section
        self.action_space = spaces.Box(self.lb_action, self.hb_action)
        self.observation_space = spaces.Box(self.lb_state, self.hb_state)

    def forward(self, state: torch.Tensor, action: torch.Tensor, by_done):
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        the shape of parameters and return must be the same as required otherwise will cause error
        when constructing your function
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :return: next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                 reward:  datatype:torch.Tensor, shape:[batch_size, 1]
                 isdone:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        warning_msg = "action out of action space!"
        if not ((action.numpy() <= self.hb_action).all() and (action.numpy() >= self.lb_action).all()):
            warnings.warn(warning_msg)
        #  define your forward function here: the format is just like: state_next = f(state,action)
        # state_next = (1+action.sum(-1))*state

        ############################################################################################
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        # reward = (state_next - state).sum(dim =-1)

        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        # isdone = torch.full(state.size()[0], True)

        ############################################################################################
        state_next = (~isdone*by_done) * state_next
        return state_next, reward, isdone

    def forward_n_step(self,func, n, state: torch.Tensor):
        reward = torch.zeros(size=[state.size()[0], n])
        isdone = state.numpy() <= self.hb_state | state.numpy() >= self.lb_state
        if np.sum(isdone) > 0:
            warning_msg = "state out of state space!"
            warnings.warn(warning_msg)
        isdone = torch.from_numpy(isdone)
        for step in range(n):
            action = func(state)
            state_next, reward[:, step], isdone = self.forward(state, action, isdone)
            state = state_next



if __name__ == "__main__":
    env = GymDemocontiModel()
