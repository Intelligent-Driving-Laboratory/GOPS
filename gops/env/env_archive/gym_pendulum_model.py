#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Acrobat Environment
#  Update Date: 2021-05-55, Wenxuan Wang: create environment


import warnings
import torch
import numpy as np

pi = torch.tensor(np.pi, dtype=torch.float32)


class GymPendulumModel(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        """
        you need to define parameters here
        """
        # define your custom parameters here
        self.max_speed = 8
        self.max_torque = 2.
        self.g = 10.0
        self.m = 1.
        self.length = 1.

        # define common parameters here
        self.state_dim = 3
        self.action_dim = 1
        lb_state = [-1., -1., -self.max_speed]
        hb_state = [1., 1., self.max_speed]
        lb_action = [-self.max_torque]
        hb_action = [self.max_torque]
        self.dt = 0.05

        # do not change the following section
        self.register_buffer('lb_state', torch.tensor(lb_state, dtype=torch.float32))
        self.register_buffer('hb_state', torch.tensor(hb_state, dtype=torch.float32))
        self.register_buffer('lb_action', torch.tensor(lb_action, dtype=torch.float32))
        self.register_buffer('hb_action', torch.tensor(hb_action, dtype=torch.float32))

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done=torch.tensor(1)):
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

        costh, sinth, thdot = state[:, 0], state[:, 1], state[:, 2]
        th = arccs(sinth, costh)
        g = self.g
        m = self.m
        length = self.length
        dt = self.dt
        newthdot = thdot + (-3 * g / (2 * length) * torch.sin(th + pi) + 3. / (m * length ** 2) * action.squeeze()) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)
        newcosth = torch.cos(newth)
        newsinth = torch.sin(newth)
        state_next = torch.stack([newcosth, newsinth, newthdot], dim=-1)
        reward = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (action ** 2).squeeze(-1)
        reward = -reward
        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = state[:, 0].new_zeros(size=[state.size()[0]], dtype= torch.bool)

        ############################################################################################
        beyond_done = beyond_done.bool()
        mask = isdone * beyond_done
        mask = torch.unsqueeze(mask, -1)
        state_next = ~mask * state_next + mask * state
        return state_next, reward, isdone


def angle_normalize(x):
    return ((x + pi) % (2 * pi)) - pi


def arccs(sinth, costh):
    eps = 0.9999  # fixme: avoid grad becomes inf when cos(theta) = 0
    th = torch.acos(eps * costh)
    th = th * (sinth > 0) + (2 * pi - th) * (sinth <= 0)
    return th


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
