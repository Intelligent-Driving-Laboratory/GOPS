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
import matplotlib.pyplot as plt

class GymMobilerobotModel:

    def __init__(self):
        """
        you need to define parameters here
        """
        self.n_obstacle = 2

        self.robot = Robot()
        self.obses = [Robot() for _ in range(self.n_obstacle)]

        # define common parameters here
        self.dt = 0.4  # seconds between state updates

        self.state_dim = (1+self.n_obstacle) * 5 + 3
        self.action_dim = 2
        self.lb_state = [-30, -30, -2*np.pi,  -1, -np.pi/2] + [-30, -np.pi, -2] + [-30, -30, -2 * np.pi,  -1, -np.pi/2] * self.n_obstacle
        self.hb_state = [30, 30, 2*np.pi,  1, np.pi/2] + [30, np.pi, 2] + [30, 30, 2 * np.pi,  1, np.pi/2] * self.n_obstacle
        self.lb_action = [-0.4, -np.pi/3]
        self.hb_action = [0.4, np.pi/3]

        # do not change the following section
        self.lb_state = torch.tensor(self.lb_state, dtype=torch.float32)
        self.hb_state = torch.tensor(self.hb_state, dtype=torch.float32)
        self.lb_action = torch.tensor(self.lb_action, dtype=torch.float32)
        self.hb_action = torch.tensor(self.hb_action, dtype=torch.float32)

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done: torch.Tensor):

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
        veh2vehdist = torch.zeros(state.shape[0], self.n_obstacle)
        for i in range(1+self.n_obstacle):
            if i == 0:
                robot_state = self.robot.f_xu(state[:, :5], action, self.dt, 'ego')
                tracking_error = self.robot.tracking_error(robot_state)
                state_next = torch.cat((robot_state, tracking_error), 1)

            else:
                obs_state = self.robot.f_xu(state[:, 3+i*5:3+i*5+5], state[:, 3+i*5+3:3+i*5+5], self.dt, 'obs')
                state_next = torch.cat((state_next, obs_state), 1)

                safe_dis = self.robot.robot_params['radius'] + self.obses[i-1].robot_params['radius'] + 0.15  # 0.35
                veh2vehdist[:,i-1] = safe_dis - (
                    torch.sqrt(torch.square(state_next[:, 3+i*5] - state_next[:, 0]) + torch.square(state_next[:, 3+i*5+1] - state_next[:, 1])))


        ############################################################################################
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        r_tracking = -1.4 * torch.square(tracking_error[:, 0]) - 1 * tracking_error[:, 1] ** 2 - 16 * tracking_error[:, 2] ** 2
        r_action = - 0.2 * action[:, 0] ** 2 - 0.5 * action[:, 1] ** 2
        reward = (r_tracking + r_action)
        ############################################################################################
        # define the constraint funtion
        constraint = veh2vehdist
        dead = veh2vehdist > 0
        info = {'constraint': constraint}
        ################################################################################################################
        # define the ending condition here the format is just like isdone = l(next_state)

        isdone = dead
        ############################################################################################

        return state_next, reward, isdone, info

    # def forward_n_step(self, func, n, state: torch.Tensor):
    #     reward = torch.zeros(size=[state.size()[0], n])
    #     isdone = state.numpy() <= self.hb_state | state.numpy() >= self.lb_state
    #     if np.sum(isdone) > 0:
    #         warning_msg = "state out of state space!"
    #         warnings.warn(warning_msg)
    #     isdone = torch.from_numpy(isdone)
    #     for step in range(n):
    #         action = func(state)
    #         state_next, reward[:, step], isdone = self.forward(state, action, isdone)
    #         state = state_next


class Robot():
    def __init__(self, path=None):
        self.robot_params = dict(v_max=0.4,
                                 w_max=np.pi / 2,
                                 v_delta_max=1.8,  # per second
                                 w_delta_max=0.8,
                                 v_desired=0.3,
                                 radius=0.74 / 2
                                 )
        self.path = path

    def f_xu(self, states, actions, T, type):
        v_delta_max = self.robot_params['v_delta_max']
        v_max = self.robot_params['v_max']
        w_max = self.robot_params['w_max']
        w_delta_max = self.robot_params['w_delta_max']
        std_type = {'ego': [0.08, 0.05], 'obs': [0.07, 0.03], 'none': [0, 0], 'explore': [0.3, 0.3]}
        stds = std_type[type]

        x, y, theta, v, w = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
        v_cmd, w_cmd = actions[:, 0], actions[:, 1]

        delta_v = torch.clamp(v_cmd - v, -v_delta_max * T, v_delta_max * T)
        delta_w = torch.clamp(w_cmd - w, -w_delta_max * T, w_delta_max * T)
        v_cmd = torch.clamp(v + delta_v, -v_max, v_max) + torch.Tensor(
            np.random.normal(0, stds[0], [states.shape[0]])) * 0.5
        w_cmd = torch.clamp(w + delta_w, -w_max, w_max) + torch.Tensor(
            np.random.normal(0, stds[1], [states.shape[0]])) * 0.5
        next_state = [x + T * torch.cos(theta) * v_cmd,
                      y + T * torch.sin(theta) * v_cmd,
                      theta + T * w_cmd,
                      v_cmd,
                      w_cmd]

        return torch.stack(next_state, 1)

    def tracking_error(self, x):
        error_position = x[:, 1]
        error_head = x[:, 2]
        # error_head = torch.where(error_head > np.pi, error_head - np.pi * 2, error_head)
        # error_head = torch.where(error_head < -np.pi, error_head + np.pi * 2, error_head)

        error_v = x[:, 3] - self.robot_params['v_desired']
        tracking = torch.cat((error_position.reshape(-1, 1), error_head.reshape(-1, 1), error_v.reshape(-1, 1)), 1)
        return tracking

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
    env = GymMobilerobotModel()
    x = torch.Tensor([[0.5297, 1.1331, -0.0566, 0.0000, 0.0000, 3.5000, -1.0000, -1.5708,
                         0.0000, 0.0000, -0.2669, -0.0566, -0.3000
                          ,
                       1.000, 1.0000, -0.2669, -0.0566, -0.3000
                       ]]*10)
    die = torch.zeros([10,2])
    env.render_init()
    for i in range(100):

        u = torch.Tensor([[0.1, 0.1]]*10)
        x, r, die, info = env.forward(x, u, die)
        env.render(x)
