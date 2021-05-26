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
        self.max_iter = None
        self.state = None

        # do not change the following section
        self.iter = 0
        self.action_space = spaces.Box(self.lb_action, self.hb_action)
        self.observation_space = spaces.Box(self.lb_state, self.hb_state)
        self.np_random, seed = seeding.np_random()

    def seed(self, seed=None):
        """
        change the random seed of the random generator in numpy
        :param seed: random seed
        :return: the seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: torch.Tensor):
        """
        rollout the model one step, notice this method will change the value of self.state
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :return: next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                 reward:  datatype:torch.Tensor, shape:[batch_size, 1]
                 done:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        self.state, reward, done = self.forward(state=self.state, action=action)
        return self.state, reward, done

    def reset(self, state=None): # TODO: batch reset
        """
        reset the state use the passed in parameter
        if no parameter is passed in, random initialize will be used
        :param state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        :return: state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        """
        if state is None:
            state = self.np_random.uniform(low=self.lb_init_state, high=self.hb_init_state, size=self.state_dim)
            self.state = torch.from_numpy(state)
        else:
            self.state = torch.from_numpy(state)
        return self.state

    def forward(self, state: torch.Tensor, action: torch.Tensor):
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
                 done:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        warning_msg = "action out of action space!"
        if not ((action.numpy() <= self.hb_action).all() and (action.numpy() >= self.lb_action).all()):
            warnings.warn(warning_msg)
        #  define your forward function here: the format is just like: state_next = f(state,action)

        ############################################################################################
        reward = self.reward(state=state_next, action=action)
        done = self.isdone(state_next)
        state_next = (~done) * state_next + done * state
        return state_next, reward, done

    def reward(self, state: torch.Tensor, action: torch.Tensor):
        """
        you need to define your own reward function here
        notice that all the variables contains the batch dim you need to remember this point
        the shape of parameters and return must be the same as required otherwise will cause error
        :param state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
        :param action:  datatype:torch.Tensor, shape:[batch_size, action_dim]
        :return: reward:  datatype:torch.Tensor, shape:[batch_size, 1]
        """
        # define the reward function here the format is just like: reward = l(state,state_next,reward)

        #############################################################################################
        return reward

    def isdone(self, next_state):
        """
        define the ending condition here
        notice that all the variables contains the batch dim you need to remember this point
        the shape of parameters and return must be the same as required otherwise will cause error
        :param next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
        :return: done:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        # define the ending condation here the format is just like done = l(next_state)

        ##############################################################################################
        if self.iter >= self.max_iter:
            done = torch.full(done.size(), True)
            self.iter = 0
       return done

if __name__ == "__main__":
    env = GymDemocontiModel()
