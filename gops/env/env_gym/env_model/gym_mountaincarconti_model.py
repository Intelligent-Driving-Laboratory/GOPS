#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Mountaincar Environment (continous, differential version)
#  Update Date: 2021-05-55, Yuhang Zhang: create environment


from typing import Tuple, Union

import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class GymMountaincarcontiModel(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None, **kwargs):
        """
        you need to define parameters here
        """
        # Define your custom parameters here
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        # Was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_position = 0.45
        self.goal_velocity = 0
        self.power = 0.0015

        # Define common parameters here
        lb_state = [self.min_position, -self.max_speed]
        hb_state = [self.max_position, self.max_speed]
        lb_action = [self.min_action]
        hb_action = [self.max_action]
        # Seconds between state updates
        self.dt = None

        super().__init__(
            obs_dim=2,
            action_dim=1,
            dt=None,
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
        #  Define your forward function here: the format is just like: state_next = f(state,action)
        pos, vec = state[:, 0], state[:, 1]
        vec = vec + self.power * action.squeeze(-1) - 0.0025 * torch.cos(3 * pos)
        vec = torch.clamp(vec, self.obs_lower_bound[1], self.obs_upper_bound[1])
        pos = pos + vec
        pos = torch.clamp(pos, self.obs_lower_bound[0], self.obs_upper_bound[0])
        vec[(pos == self.obs_lower_bound[0]) & (vec < 0)] = 0
        state_next = torch.stack([pos, vec], dim=-1)

        ############################################################################################
        # Sefine the ending condation here the format is just like isdone = l(next_state)
        isdone = (pos >= self.goal_position) & (vec >= self.goal_velocity)

        ############################################################################################
        # Define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = torch.zeros(state.size()[0])
        reward[isdone] = 100.0
        reward = reward - 0.1 * action.squeeze(-1) ** 2

        return state_next, reward, isdone, {}
