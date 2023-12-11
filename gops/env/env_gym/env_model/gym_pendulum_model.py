#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Acrobat Environment
#  Update Date: 2021-05-55, Wenxuan Wang: create environment


from typing import Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict
from gops.utils.math_utils import angle_normalize

pi = torch.tensor(np.pi, dtype=torch.float32)


class GymPendulumModel(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None, **kwargs):
        """
        you need to define parameters here
        """
        # Define your custom parameters here
        self.max_speed = 8
        self.max_torque = 2.0
        self.g = 10.0
        self.m = 1.0
        self.length = 1.0

        # Define common parameters here
        lb_state = [-1.0, -1.0, -self.max_speed]
        hb_state = [1.0, 1.0, self.max_speed]
        lb_action = [-self.max_torque]
        hb_action = [self.max_torque]

        super().__init__(
            obs_dim=3,
            action_dim=1,
            dt=0.05,
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
        :param obs: datatype:torch.Tensor, shape:[batch_size, state_dim]
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :param done: flag indicate the state is already done which means it will not be calculated by the model
        :return:
                next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                reward:  datatype:torch.Tensor, shape:[batch_size, 1]
                isdone:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        state = obs
        costh, sinth, thdot = state[:, 0], state[:, 1], state[:, 2]
        th = arccs(sinth, costh)
        g = self.g
        m = self.m
        length = self.length
        dt = self.dt
        newthdot = (
            thdot
            + (
                -3 * g / (2 * length) * torch.sin(th + pi)
                + 3.0 / (m * length ** 2) * action.squeeze()
            )
            * dt
        )
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -self.max_speed, self.max_speed)
        newcosth = torch.cos(newth)
        newsinth = torch.sin(newth)
        state_next = torch.stack([newcosth, newsinth, newthdot], dim=-1)
        reward = (
            angle_normalize(th) ** 2
            + 0.1 * thdot ** 2
            + 0.001 * (action ** 2).squeeze(-1)
        )
        reward = -reward
        ############################################################################################

        # Define the ending condation here the format is just like isdone = l(next_state)
        isdone = state[:, 0].new_zeros(size=[state.size()[0]], dtype=torch.bool)

        return state_next, reward, isdone, {}


def arccs(sinth, costh):
    eps = 0.9999  # fixme: avoid grad becomes inf when cos(theta) = 0
    th = torch.acos(eps * costh)
    th = th * (sinth > 0) + (2 * pi - th) * (sinth <= 0)
    return th
