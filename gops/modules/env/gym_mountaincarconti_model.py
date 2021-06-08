#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Mountaincar Environment (continous, differential version)
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import warnings
import torch
import numpy as np

pi = torch.tensor(np.pi, dtype=torch.float32)


class GymMountaincarcontiModel(object):
    def __init__(self):
        """
        you need to define parameters here
        """
        # define your custom parameters here
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = 0
        self.power = 0.0015

        # define common parameters here
        self.state_dim = 2
        self.action_dim = 1
        self.lb_state = [self.min_position, -self.max_speed]
        self.hb_state = [self.max_position, self.max_speed]
        self.lb_action = [self.min_action]
        self.hb_action = [self.max_action]
        self.dt = None  # seconds between state updates

        # do not change the following section
        self.lb_state = torch.tensor(self.lb_state, dtype=torch.float32)
        self.hb_state = torch.tensor(self.hb_state, dtype=torch.float32)
        self.lb_action = torch.tensor(self.lb_action, dtype=torch.float32)
        self.hb_action = torch.tensor(self.hb_action, dtype=torch.float32)

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done):
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        when constructing your function
        :param state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :param beyond_done: flag indicate the state is already done which means it will not be calculated by the model
        :return:
                next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                reward:  datatype:torch.Tensor, shape:[batch_size, 1]
                isdone:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        warning_msg = "action out of action space!"
        if not ((action <= self.hb_action).all() and (action >= self.lb_action).all()):
            warnings.warn(warning_msg)
            action = clip_by_tensor(action, self.lb_action, self.hb_action)

        warning_msg = "state out of state space!"
        if not ((state <= self.hb_state).all() and (state >= self.lb_state).all()):
            warnings.warn(warning_msg)
            state = clip_by_tensor(state, self.lb_state, self.hb_state)

        #  define your forward function here: the format is just like: state_next = f(state,action)
        pos, vec = state[:, 0], state[:, 1]
        vec = vec + self.power*action.squeeze(-1) - 0.0025*torch.cos(3*pos)
        vec = clip_by_tensor(vec, self.lb_state[1], self.hb_state[1])
        pos = pos + vec
        pos = clip_by_tensor(pos, self.lb_state[0], self.hb_state[0])
        vec[(pos == self.lb_state[0]) & (vec < 0)] = 0
        state_next = torch.stack([pos, vec], dim=-1)

        ############################################################################################
        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = (pos >= self.goal_position) & (vec >= self.goal_velocity)

        ############################################################################################
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = torch.zeros(state.size()[0])
        reward[isdone] = 100.0
        reward = reward - 0.1*action.squeeze(-1)**2

        ############################################################################################

        beyond_done = beyond_done.bool()
        mask = isdone * beyond_done
        mask = torch.unsqueeze(mask, -1)
        state_next = ~mask * state_next + mask * state
        return state_next, reward, isdone

    def forward_n_step(self, func, n, state: torch.Tensor):
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


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result