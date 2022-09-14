#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab

from gops.env.resources.intersection.dynamics_and_models import EnvironmentModel

import warnings

import numpy as np
import torch


class PythIntersectionModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        """
        you need to define parameters here
        """
        # define your custom parameters here

        self.dynamics = EnvironmentModel("left")

        # define common parameters here
        self.state_dim = 49
        self.action_dim = 2
        self.constraint_dim = 32
        self.use_constraint = kwargs.get("use_constraint", True)
        self.lb_state = [-np.inf] * self.state_dim
        self.hb_state = [np.inf] * self.state_dim
        self.lb_action = [-1.0, -1.0]
        self.hb_action = [1.0, 1.0]
        self.dt = 0.1  # seconds between state updates

        # do not change the following section
        self.lb_state = torch.tensor(self.lb_state, dtype=torch.float32)
        self.hb_state = torch.tensor(self.hb_state, dtype=torch.float32)
        self.lb_action = torch.tensor(self.lb_action, dtype=torch.float32)
        self.hb_action = torch.tensor(self.hb_action, dtype=torch.float32)

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
        ego_state = state[:, :9]
        veh_state = state[:, 9:]

        (
            obses_ego,
            obses_veh,
            rewards,
            punish_term_for_training,
            real_punish_term,
            veh2veh4real,
            constraints
        ) = self.dynamics.rollout_out(ego_state, veh_state, action)

        state_next = torch.cat([ego_state, veh_state], 1)

        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = torch.full([state.size()[0]], False, dtype=torch.float32)

        ############################################################################################
        info = {"constraint": constraints}

        return state_next, rewards, isdone, info

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


def env_model_creator(**kwargs):
    """
    make env model `pyth_intersection`
    """
    return PythIntersectionModel(**kwargs)


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
