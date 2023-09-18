#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Acrobat Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment


import math
from typing import Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class GymCartpolecontiModel(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None, **kwargs):
        """
        you need to define parameters here
        """
        # Define your custom parameters here
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        # Actually half the pole's length
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        # 12deg
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.max_x = self.x_threshold * 2
        self.min_x = -self.max_x
        self.max_x_dot = np.finfo(np.float32).max
        self.min_x_dot = -np.finfo(np.float32).max
        # 24deg
        self.max_theta = self.theta_threshold_radians * 2
        self.min_theta = -self.max_theta
        self.max_theta_dot = np.finfo(np.float32).max
        self.min_theta_dot = -np.finfo(np.float32).max
        self.min_action = -1.0
        self.max_action = 1.0

        # Define common parameters here
        lb_state = [self.min_x, self.min_x_dot, self.min_theta, self.min_theta_dot]
        hb_state = [self.max_x, self.max_x_dot, self.max_theta, self.max_theta_dot]
        lb_action = [self.min_action]
        hb_action = [self.max_action]

        super().__init__(
            obs_dim=4,
            action_dim=1,
            dt=0.02,
            obs_lower_bound=lb_state,
            obs_upper_bound=hb_state,
            action_lower_bound=lb_action,
            action_upper_bound=hb_action,
            device=device,
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        when constructing your function
        :param obs: datatype:torch.Tensor, shape:[batch_size, obs_dim]
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :param done: flag indicate the state is already done which means it will not be calculated by the model
        :param info: datatype: InfoDict, any useful information for debug or training, including constraint,
                     adversary action, etc
        :return:
                next_obs:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                             the state will not change anymore when the corresponding flag done is set to True
                reward:  datatype:torch.Tensor, shape:[batch_size,]
                isdone:   datatype:torch.Tensor, shape:[batch_size,]
                          flag done will be set to true when the model reaches the max_iteration or the next state
                          satisfies ending condition

                info: datatype: InfoDict, any useful information for debug or training, including constraint,
                      adversary action, etc
        """
        state = obs
        #  Define your forward function here: the format is just like: state_next = f(state,action)
        x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        force = self.force_mag * action
        temp = (
            torch.squeeze(force)
            + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        state_next = torch.stack([x, x_dot, theta, theta_dot]).transpose(1, 0)
        ################################################################################################################
        # Define the ending condation here the format is just like isdone = l(next_state)
        isdone = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )
        ############################################################################################
        # Define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = 1 - isdone.float()

        return state_next, reward, isdone, {"state": state_next}


def env_model_creator(**kwargs):
    return GymCartpolecontiModel(**kwargs)
