#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab

import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np


class Dynamics(object):
    def __init__(self):
        self.mass_cart = 9.42477796
        self.mass_rod1 = 4.1033127
        self.mass_rod2 = 4.1033127
        self.l_rod1 = 0.6
        self.l_rod2 = 0.6
        self.g = 9.81
        self.damping_cart = 0.0
        self.damping_rod1 = 0.0
        self.damping_rod2 = 0.0

    def f_xu(self, states, actions, tau):
        m, m1, m2 = self.mass_cart, self.mass_rod1, self.mass_rod2

        l1, l2 = self.l_rod1, self.l_rod2

        d1, d2, d3 = self.damping_cart, self.damping_rod1, self.damping_rod2

        g = self.g

        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )

        u = actions[:, 0]

        ones = torch.ones_like(p, dtype=torch.float32)
        M = torch.stack(
            [
                (m + m1 + m2) * ones,
                l1 * (0.5 * m1 + m2) * torch.cos(theta1),
                0.5 * m2 * l2 * torch.cos(theta2),
                l1 * (0.5 * m1 + m2) * torch.cos(theta1),
                l1 * l1 * (0.3333 * m1 + m2) * ones,
                0.5 * l1 * l2 * m2 * torch.cos(theta1 - theta2),
                0.5 * l2 * m2 * torch.cos(theta2),
                0.5 * l1 * l2 * m2 * torch.cos(theta1 - theta2),
                0.3333 * l2 * l2 * m2 * ones,
            ],
            dim=1,
        ).reshape(-1, 3, 3)

        f = torch.stack(
            [
                l1 * (0.5 * m1 + m2) * torch.square(theta1dot) * torch.sin(theta1)
                + 0.5 * m2 * l2 * torch.square(theta2dot) * torch.sin(theta2)
                - d1 * pdot
                + u,
                -0.5 * l1 * l2 * m2 * torch.square(theta2dot) * torch.sin(theta1 - theta2)
                + g * (0.5 * m1 + m2) * l1 * torch.sin(theta1)
                - d2 * theta1dot,
                0.5 * l1 * l2 * m2 * torch.square(theta1dot) * torch.sin(theta1 - theta2)
                + g * 0.5 * l2 * m2 * torch.sin(theta2),
            ],
            dim=1,
        ).reshape(-1, 3, 1)

        M_inv = torch.linalg.inv(M)
        tmp = torch.matmul(M_inv, f).squeeze(-1)

        deriv = torch.cat([states[:, 3:], tmp], dim=-1)
        next_states = states + tau * deriv

        return next_states

    def compute_rewards(self, states):  # obses and actions are tensors

        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )
        tip_x = p + self.l_rod1 * torch.sin(theta1) + self.l_rod2 * torch.sin(theta2)
        tip_y = self.l_rod1 * torch.cos(theta1) + self.l_rod2 * torch.cos(theta2)
        dist_penalty = 0.01 * torch.square(tip_x) + torch.square(tip_y - 2)
        v1, v2 = theta1dot, theta2dot
        vel_penalty = 1e-3 * torch.square(v1) + 5e-3 * torch.square(v2)
        rewards = 10 - dist_penalty - vel_penalty

        return rewards

    def get_done(self, states):
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )
        point0x, point0y = p, 0
        point1x, point1y = point0x + self.l_rod1 * np.sin(
            theta1
        ), point0y + self.l_rod1 * np.cos(theta1)
        point2x, point2y = point1x + self.l_rod2 * np.sin(
            theta2
        ), point1y + self.l_rod2 * np.cos(theta2)

        return point2y <= 1.0

class PythInvertedpendulum(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        """
        you need to define parameters here
        """
        # define your custom parameters here
        self.dynamics = Dynamics()
        # define common parameters here
        self.state_dim = 6
        self.action_dim = 1
        lb_state = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        hb_state = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        lb_action = [-1.0]
        hb_action = [1.0]
        self.dt = 0.01  # seconds between state updates

        # do not change the following section
        self.lb_state = torch.tensor(lb_state, dtype=torch.float32)
        self.hb_state = torch.tensor(hb_state, dtype=torch.float32)
        self.lb_action = torch.tensor(lb_action, dtype=torch.float32)
        self.hb_action = torch.tensor(hb_action, dtype=torch.float32)

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done=None):
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
                reward:  datatype:torch.Tensor, shape:[batch_size,]
                isdone:   datatype:torch.Tensor, shape:[batch_size,]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition

                info: datatype: dict, any useful information for debug or training, including constraint
                        {"constraint": None}
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
        state_next = self.dynamics.f_xu(state, 500 * action, self.dt)

        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = self.dynamics.get_done(state_next)

        ############################################################################################

        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = self.dynamics.compute_rewards(state_next)

        ############################################################################################
        if beyond_done is None:
            beyond_done = torch.full([state.size()[0]], False, dtype=torch.float32)

        beyond_done = beyond_done.bool()
        mask = isdone | beyond_done
        mask = torch.unsqueeze(mask, -1)
        state_next = ~mask * state_next + mask * state
        reward = ~(beyond_done) * reward
        return state_next, reward, mask.squeeze(), {}

    def forward_n_step(self, func, n, state: torch.Tensor):
        raise NotImplementedError


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


def env_model_creator(**kwargs):
    """
    make env model `pyth_invertedpendulum`
    """
    return PythInvertedpendulum(**kwargs)



