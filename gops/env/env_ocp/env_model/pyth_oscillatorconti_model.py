#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Oscillator Model
#  Update Date: 2022-08-12, Jie Li: create environment
#  Update Date: 2022-10-24, Yujie Yang: add wrapper

from typing import Tuple, Union

import torch
import numpy as np

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict

pi = torch.tensor(np.pi, dtype=torch.float32)


class PythOscillatorcontiModel(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None, **kwargs):
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs["is_adversary"]
        self.sample_batch_size = kwargs["reset_batch_size"]
        self.state_dim = 2
        self.action_dim = 1
        self.adversary_dim = 1
        self.dt = 1 / 200

        # utility information
        self.Q = torch.eye(self.state_dim)
        self.R = torch.eye(self.action_dim)
        self.gamma = 1
        self.gamma_atte = kwargs["gamma_atte"]

        # state & action space
        self.fixed_initial_state = kwargs["fixed_initial_state"]
        self.initial_state_range = kwargs["initial_state_range"]
        self.battery_a_initial = self.initial_state_range[0]
        self.battery_b_initial = self.initial_state_range[1]
        self.state_threshold = kwargs["state_threshold"]
        self.battery_a_threshold = self.state_threshold[0]
        self.battery_b_threshold = self.state_threshold[1]
        self.min_action = [-5.0]
        self.max_action = [5.0]
        self.min_adv_action = [-1.0 / self.gamma_atte]
        self.max_adv_action = [1.0 / self.gamma_atte]

        self.lb_state = torch.tensor(
            [-self.battery_a_threshold, -self.battery_b_threshold], dtype=torch.float32
        )
        self.hb_state = torch.tensor(
            [self.battery_a_threshold, self.battery_b_threshold], dtype=torch.float32
        )
        if self.is_adversary:
            self.lb_action = torch.tensor(
                self.min_action + self.min_adv_action, dtype=torch.float32
            )
            self.hb_action = torch.tensor(
                self.max_action + self.max_adv_action, dtype=torch.float32
            )
        else:
            self.lb_action = torch.tensor(self.min_action, dtype=torch.float32)
            self.hb_action = torch.tensor(self.max_action, dtype=torch.float32)

        self.ones_ = torch.ones(self.sample_batch_size)
        self.zeros_ = torch.zeros(self.sample_batch_size)

        # parallel sample
        self.parallel_state = None
        self.lower_step = kwargs["lower_step"]
        self.upper_step = kwargs["upper_step"]
        self.max_step_per_episode = self.max_step()
        self.step_per_episode = self.initial_step()

        super().__init__(
            obs_dim=self.state_dim,
            action_dim=self.action_dim,
            dt=self.dt,
            obs_lower_bound=self.lb_state,
            obs_upper_bound=self.hb_state,
            action_lower_bound=self.lb_action,
            action_upper_bound=self.hb_action,
            device=device,
        )

    def max_step(self):
        return torch.from_numpy(
            np.floor(
                np.random.uniform(
                    self.lower_step, self.upper_step, [self.sample_batch_size]
                )
            )
        )

    def initial_step(self):
        return torch.zeros(self.sample_batch_size)

    def reset(self):

        battery_a = np.random.uniform(
            -self.battery_a_initial, self.battery_a_initial, [self.sample_batch_size, 1]
        )
        battery_b = np.random.uniform(
            -self.battery_b_initial, self.battery_b_initial, [self.sample_batch_size, 1]
        )

        state = np.concatenate([battery_a, battery_b], axis=1)

        return torch.from_numpy(state).float()

    def step(self, action: torch.Tensor):
        dt = self.dt
        battery_a, battery_b = self.parallel_state[:, 0], self.parallel_state[:, 1]
        # memristor
        memristor = action[:, 0]
        # noise
        noise = action[:, 1]

        deri_battery_a = -0.25 * battery_a
        deri_battery_b = (
            0.5 * torch.mul(battery_a ** 2, battery_b)
            - 1 / (2 * self.gamma_atte ** 2) * battery_b ** 3
            - 0.5 * battery_b
            + torch.mul(battery_a, memristor)
            + torch.mul(battery_b, noise)
        )

        delta_state = torch.stack([deri_battery_a, deri_battery_b], dim=-1)
        self.parallel_state = self.parallel_state + delta_state * dt

        reward = (
            self.Q[0][0] * battery_a ** 2
            + self.Q[1][1] * battery_b ** 2
            + self.R[0][0] * (memristor ** 2).squeeze(-1)
            - self.gamma_atte ** 2 * (noise ** 2).squeeze(-1)
        )

        # define the ending condation here the format is just like isdone = l(next_state)
        done = (
            torch.where(
                abs(self.parallel_state[:, 0]) > self.battery_a_threshold,
                self.ones_,
                self.zeros_,
            ).bool()
            | torch.where(
                abs(self.parallel_state[:, 1]) > self.battery_b_threshold,
                self.ones_,
                self.zeros_,
            ).bool()
        )

        self.step_per_episode += 1
        info = {
            "TimeLimit.truncated": torch.where(
                self.step_per_episode > self.max_step_per_episode,
                self.ones_,
                self.zeros_,
            ).bool()
        }

        return self.parallel_state, reward, done, info

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

        dt = self.dt
        battery_a, battery_b = state[:, 0], state[:, 1]
        # memristor
        memristor = action[:, 0]
        if self.is_adversary:
            # noise
            noise = action[:, 1]
        else:
            noise = torch.zeros_like(memristor)

        deri_battery_a = -0.25 * battery_a
        deri_battery_b = (
            0.5 * torch.mul(battery_a ** 2, battery_b)
            - 1 / (2 * self.gamma_atte ** 2) * battery_b ** 3
            - 0.5 * battery_b
            + torch.mul(battery_a, memristor)
            + torch.mul(battery_b, noise)
        )

        delta_state = torch.stack([deri_battery_a, deri_battery_b], dim=-1)
        state_next = state + delta_state * dt
        cost = (
            self.Q[0][0] * battery_a ** 2
            + self.Q[1][1] * battery_b ** 2
            + self.R[0][0] * (memristor ** 2).squeeze(-1)
            - self.gamma_atte ** 2 * (noise ** 2).squeeze(-1)
        )
        reward = -cost
        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = state[:, 0].new_zeros(size=[state.size()[0]], dtype=torch.bool)

        ############################################################################################
        # beyond_done = beyond_done.bool()
        # mask = isdone * beyond_done
        # mask = torch.unsqueeze(mask, -1)
        # state_next = ~mask * state_next + mask * state
        return state_next, reward, isdone, {"delta_state": delta_state}

    def f_x(self, state, batch_size):

        if batch_size > 1:
            fx = torch.zeros((batch_size, self.state_dim))
            fx[:, 0] = -0.25 * state[:, 0]
            fx[:, 1] = (
                0.5 * torch.mul(state[:, 0] ** 2, state[:, 1])
                - 1 / (2 * self.gamma_atte ** 2) * state[:, 1] ** 3
                - 0.5 * state[:, 1]
            )
        else:
            fx = torch.zeros((self.state_dim, 1))
            fx[0, 0] = -0.25 * state[0, 0]
            fx[1, 0] = (
                0.5 * torch.mul(state[0, 0] ** 2, state[0, 1])
                - 1 / (2 * self.gamma_atte ** 2) * state[0, 1] ** 3
                - 0.5 * state[0, 1]
            )

        return fx

    def g_x(self, state, batch_size):

        if batch_size > 1:
            gx = torch.zeros((batch_size, self.state_dim, self.action_dim))
            gx[:, 0, 0] = torch.zeros((batch_size,))
            gx[:, 1, 0] = state[:, 0]
        else:
            gx = torch.zeros((self.state_dim, self.action_dim))
            gx[0, 0] = 0
            gx[1, 0] = state[0, 0]

        return gx

    def best_act(self, state, delta_value):
        batch_size = state.size()[0]

        if batch_size > 1:
            gx = self.g_x(state, batch_size)
            delta_value = delta_value[:, :, np.newaxis]
            act = -0.5 * torch.matmul(
                self.R.inverse(), torch.bmm(gx.transpose(1, 2), delta_value)
            ).squeeze(-1)
        else:
            gx = self.g_x(state, batch_size)
            act = -0.5 * torch.mm(self.R.inverse(), torch.mm(gx.t(), delta_value.t()))

        return act.detach()

    def k_x(self, state, batch_size):

        if batch_size > 1:
            kx = torch.zeros((batch_size, self.state_dim, self.adversary_dim))
            kx[:, 0, 0] = torch.zeros((batch_size,))
            kx[:, 1, 0] = state[:, 1]
        else:
            kx = torch.zeros((self.state_dim, self.adversary_dim))
            kx[0, 0] = 0
            kx[1, 0] = state[0, 1]

        return kx

    def worst_adv(self, state, delta_value):
        batch_size = state.size()[0]

        if batch_size > 1:
            kx = self.k_x(state, batch_size)
            delta_value = delta_value[:, :, np.newaxis]
            adv = (
                0.5
                / (self.gamma_atte ** 2)
                * torch.bmm(kx.transpose(1, 2), delta_value).squeeze(-1)
            )
        else:
            kx = self.k_x(state, batch_size)
            adv = 0.5 / (self.gamma_atte ** 2) * torch.mm(kx.t(), delta_value.t())

        return adv.detach()
