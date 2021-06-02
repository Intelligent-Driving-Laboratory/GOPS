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

pi = torch.from_numpy(np.array([np.pi],dtype=np.float32))
class GymPendulumModel:
    def __init__(self):
        """
        you need to define parameters here
        """
        # define your custom parameters here
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.
        self.l = 1.
        self.pi = torch.from_numpy(np.array([np.pi],dtype=np.float32))
        # define common parameters here
        self.state_dim = 3
        self.action_dim = 1
        self.lb_state = -np.array([1., 1., self.max_speed], dtype=np.float32)
        self.hb_state = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.lb_action = -np.array([self.max_torque], dtype=np.float32)
        self.hb_action = np.array([self.max_torque])
        self.lb_init_state = -np.array([1., 1., self.max_speed], dtype=np.float32)
        self.hb_init_state = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.tau = 0.02    # seconds between state updates   #  dt
        self.max_iter = 1000
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
        if not ((action <= torch.from_numpy(self.hb_action)).all() and
                (action >= torch.from_numpy(self.lb_action)).all()):
            warnings.warn(warning_msg)
        #  define your forward function here: the format is just like: state_next = f(state,action)
        costh, sinth, thdot = state[:,0:1], state[:,1:2], state[:,2:]
        th = arccs(sinth, costh)
        g = self.g
        m = self.m
        l = self.l
        dt = self.tau
        pi = self.pi
        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + pi) + 3. / (m * l ** 2) * action) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)
        newcosth = torch.cos(newth)
        newsinth = torch.sin(newth)
        state_next = torch.cat([newcosth, newsinth, newthdot], dim=1)
        ############################################################################################
        reward = self.reward(state=state_next, action=action)
        isdone = self.isdone(state_next)
        state_next = (~isdone) * state_next + isdone * state
        return state_next, reward, isdone

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
        costh, sinth, thdot = state[:,0:1], state[:,1:2], state[:,2:]
        th = arccs(sinth, costh)
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (action ** 2)
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
        done = torch.full([next_state.size()[0], 1], False, dtype=torch.bool)
        ##############################################################################################
        if self.iter >= self.max_iter:
            done = torch.full([next_state.size()[0],1], True, dtype=torch.bool)
            self.iter = 0
        return done


def angle_normalize(x):
    return ((x + pi) % (2 * pi)) - pi


def arccs(sinth, costh):
    eps= 0.9999
    th = torch.acos(eps*costh)
    th = th*(sinth > 0) +(2*pi - th)*(sinth <= 0)
    return th
