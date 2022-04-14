#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment


import math
import warnings
import numpy as np
import torch
import copy


class SimuVehicle3dofcontiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        you need to define or revise parameters here
        """
        # define your custom parameters here
        self.mass = 2000  # kg
        self.Izz = 4000  # kg*m^2
        self.lf = 1.4  # m
        self.lr = 1.6  # m
        self.kf = -12000  # N/rad
        self.kr = -11000  # N/rad

        self.min_x = -9999
        self.max_x = 9999
        self.min_y = -9999
        self.max_y = 9999
        self.min_u = -9999
        self.max_u = 9999
        self.min_v = -9999
        self.max_v = 9999
        self.min_phi = -np.pi
        self.max_phi = np.pi
        self.min_omega = -np.pi
        self.max_omega = np.pi
        self.min_steel = -1.5
        self.max_steel = 1.5
        self.min_force = -9999
        self.max_force = 9999

        # define common parameters here
        self.dt = 0.02  # seconds between state updates
        self.state_dim = 6
        self.action_dim = 2
        lb_state = [
            self.min_x,
            self.min_y,
            self.min_u,
            self.min_v,
            self.min_phi,
            self.min_omega,
        ]
        hb_state = [
            self.max_x,
            self.max_y,
            self.max_u,
            self.max_v,
            self.max_phi,
            self.max_omega,
        ]
        lb_action = [
            self.min_steel,
            self.min_force,
            self.min_force,
            self.min_force,
            self.min_force,
        ]
        hb_action = [
            self.max_steel,
            self.max_force,
            self.max_force,
            self.max_force,
            self.max_force,
        ]

        # do not change the following section

        self.register_buffer("lb_state", torch.tensor(lb_state, dtype=torch.float32))
        self.register_buffer("hb_state", torch.tensor(hb_state, dtype=torch.float32))
        self.register_buffer("lb_action", torch.tensor(lb_action, dtype=torch.float32))
        self.register_buffer("hb_action", torch.tensor(hb_action, dtype=torch.float32))

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
        state_next = state
        act_steer = action[:, 0]
        act_acc = (action[:, 1] * torch.cos(act_steer) + action[:, 2] + action[:, 3] + action[:, 4]) / self.mass

        state_next[:, 0] = state[:, 0] + self.dt * (
            state[:, 2] * torch.cos(state[:, 4]) - state[:, 3] * torch.sin(state[:, 4])
        )  # "X"
        state_next[:, 1] = state[:, 1] + self.dt * (
            state[:, 3] * torch.cos(state[:, 4]) + state[:, 2] * torch.sin(state[:, 4])
        )  # "Y"
        state_next[:, 2] = state[:, 2] + self.dt * act_acc  # "u"
        state_next[:, 3] = (
            self.mass * state[:, 2] * state[:, 3]
            + self.dt * (self.lf * self.kf - self.lr * self.kr) * state[:, 5]
            - self.dt * self.kf * act_steer * state[:, 2]
            - self.dt * self.mass * (state[:, 2] ** 2) * state[:, 5]
        ) / (
            self.mass * state[:, 2] - self.dt * (self.kf + self.kr)
        )  # "v"
        state_next[:, 4] = state[:, 4] + self.dt * state[:, 5]  # "Phi"
        state_next[:, 5] = (
            self.Izz * state[:, 2] * state[:, 5]
            + self.dt * (self.lf * self.kf - self.lr * self.kr) * state[:, 3]
            - self.dt * self.lf * self.kf * act_steer * state[:, 2]
        ) / (
            self.Izz * state[:, 2] - self.dt * (self.lf * self.lf * self.kf + self.lr * self.lr * self.kr)
        )  # "r"
        # state_next = state_next.transpose(0, 1)

        ################################################################################################################
        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = (
            (state_next[:, 0] < self.min_x)
            + (state_next[:, 0] > self.max_x)
            + (state_next[:, 1] < self.min_y)
            + (state_next[:, 1] > self.max_y)
            + (state_next[:, 2] < self.min_u)
            + (state_next[:, 2] > self.max_u)
            + (state_next[:, 3] < self.min_v)
            + (state_next[:, 3] > self.max_v)
            + (state_next[:, 4] < self.min_phi)
            + (state_next[:, 4] > self.max_phi)
            + (state_next[:, 5] < self.min_omega)
            + (state_next[:, 5] > self.max_omega)
        )

        ############################################################################################
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = (
            -pow((state_next[:, 2] - 5), 2)
            - pow((state_next[:, 1] - 3 * torch.sin(state_next[:, 0])), 2)
            - pow((0.1 * action[:, 0] + 0.001 * action[:, 1]), 2)
        )
        ############################################################################################
        return state_next, reward, isdone, {}

    def forward_n_step(self, func, n, state: torch.Tensor):
        reward = torch.zeros(size=[state.size()[0], n])
        isdone = state.numpy() <= self.hb_state | state.numpy() >= self.lb_state
        if np.sum(isdone) > 0:
            warning_msg = "state out of state space!"
            warnings.warn(warning_msg)
        isdone = torch.from_numpy(isdone)
        for step in range(n):
            action = func(state)
            state_next, reward[:, step], isdone, _ = self.forward(state, action, isdone)
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
    e = SimuVehicle3dofcontiModel()
    state = torch.tensor([[0.0, 0.0, 0.0001, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0001, 0.0, 0.0, 0.0]])
    reward = torch.zeros(size=[state.size()[0], 1000])

    def func(action):
        return action

    res = []
    for step in range(1000):
        # steer = np.pi*10 / 180 * math.sin(step * 0.005 * math.pi)
        # acc = 1000
        steer = np.pi * 5 / 180
        acc = 1000 * math.sin(step * 0.005 * math.pi) + 1000

        action = torch.tensor([[steer, 0.0, acc, 0.0, 0.0], [steer, 0.0, acc, 0.0, 0.0]])
        state_next, reward[:, step], isdone = e.forward(state, action)
        res.append(copy.deepcopy(state_next.numpy()[0]))
        state = state_next
    np.savetxt("accuracy_compare/sin_force_1000_step_5deg_rear.csv", res)
