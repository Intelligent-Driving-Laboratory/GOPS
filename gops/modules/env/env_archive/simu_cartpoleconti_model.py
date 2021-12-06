#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Acrobat Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import math
import warnings
import numpy as np
import torch


class SimuCartpolecontiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        you need to define parameters here
        """
        # define your custom parameters here
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 12deg
        self.x_threshold = 2.4
        self.max_x = self.x_threshold * 2
        self.min_x = -self.max_x
        self.max_x_dot = np.finfo(np.float32).max
        self.min_x_dot = -np.finfo(np.float32).max
        self.max_theta = self.theta_threshold_radians * 2  # 24deg
        self.min_theta = -self.max_theta
        self.max_theta_dot = np.finfo(np.float32).max
        self.min_theta_dot = -np.finfo(np.float32).max
        self.min_action = -1.0
        self.max_action = 1.0

        # define common parameters here
        self.dt = 0.02  # seconds between state updates
        self.state_dim = 4
        self.action_dim = 1
        lb_state = [self.min_x, self.min_x_dot, self.min_theta,  self.min_theta_dot]
        hb_state = [self.max_x, self.max_x_dot, self.max_theta,  self.max_theta_dot]
        lb_action = [self.min_action]
        hb_action = [self.max_action]

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
        ################################################################################################################
        #  define your forward function here: the format is just like: state_next = f(state,action)
        x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        force = self.force_mag * action
        temp = (torch.squeeze(force) + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        state_next = torch.stack([x, x_dot, theta, theta_dot]).transpose(1, 0)
        ################################################################################################################
        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = (x < -self.x_threshold) + \
                 (x > self.x_threshold) + \
                 (theta < -self.theta_threshold_radians) + \
                 (theta > self.theta_threshold_radians)
        ############################################################################################
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = 1 - isdone.float()
        ############################################################################################

        beyond_done = beyond_done.bool()
        mask = isdone * beyond_done
        mask = torch.unsqueeze(mask, -1)
        state_next = ~mask * state_next + mask * state
        reward = ~(isdone * beyond_done) * reward
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


if __name__ == "__main__":
    pass
