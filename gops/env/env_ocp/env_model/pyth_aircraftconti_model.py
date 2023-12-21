#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Aircraft Model
#  Update Date: 2022-08-12, Jie Li: create environment
#  Update Date: 2022-10-24, Yujie Yang: add wrapper

from typing import Tuple, Union

import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class PythAircraftcontiModel(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None, **kwargs):
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs["is_adversary"]
        self.sample_batch_size = kwargs["reset_batch_size"]
        self.state_dim = 3
        self.action_dim = 1
        self.adversary_dim = 1
        self.dt = 1 / 200  # seconds between state updates

        # define your custom parameters here
        self.A = torch.tensor(
            [[-1.01887, 0.90506, -0.00215], [0.82225, -1.07741, -0.17555], [0, 0, -1]],
            dtype=torch.float32,
        )
        self.A_attack_ang = torch.tensor(
            [-1.01887, 0.90506, -0.00215], dtype=torch.float32
        ).reshape((3, 1))
        self.A_rate = torch.tensor(
            [0.82225, -1.07741, -0.17555], dtype=torch.float32
        ).reshape((3, 1))
        self.A_elevator_ang = torch.tensor([0, 0, -1], dtype=torch.float32).reshape(
            (3, 1)
        )
        self.B = torch.tensor([0.0, 0.0, 1.0]).reshape((3, 1))
        self.D = torch.tensor([1.0, 0.0, 0.0]).reshape((3, 1))

        # utility information
        self.Q = torch.eye(self.state_dim)
        self.R = torch.eye(self.action_dim)
        self.gamma = 1
        self.gamma_atte = kwargs["gamma_atte"]

        # state & action space
        self.fixed_initial_state = kwargs["fixed_initial_state"]
        self.initial_state_range = kwargs["initial_state_range"]
        self.attack_ang_initial = self.initial_state_range[0]
        self.rate_initial = self.initial_state_range[1]
        self.elevator_ang_initial = self.initial_state_range[2]
        self.state_threshold = kwargs["state_threshold"]
        self.attack_ang_threshold = self.state_threshold[0]
        self.rate_threshold = self.state_threshold[1]
        self.elevator_ang_threshold = self.state_threshold[2]
        self.max_action = [1.0]
        self.min_action = [-1.0]
        self.max_adv_action = [1.0 / self.gamma_atte]
        self.min_adv_action = [-1.0 / self.gamma_atte]

        self.lb_state = torch.tensor(
            [
                -self.attack_ang_threshold,
                -self.rate_threshold,
                -self.elevator_ang_threshold,
            ],
            dtype=torch.float32,
        )
        self.hb_state = torch.tensor(
            [
                self.attack_ang_threshold,
                self.rate_threshold,
                self.elevator_ang_threshold,
            ],
            dtype=torch.float32,
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

        attack_ang = np.random.normal(
            0, self.attack_ang_initial, [self.sample_batch_size, 1]
        )
        rate = np.random.normal(0, self.rate_initial, [self.sample_batch_size, 1])
        elevator_ang = np.random.normal(
            0, self.elevator_ang_initial, [self.sample_batch_size, 1]
        )

        state = np.concatenate([attack_ang, rate, elevator_ang], axis=1)

        return torch.from_numpy(state).float()

    def step(self, action: torch.Tensor):
        dt = self.dt
        A_attack_ang = self.A_attack_ang
        A_rate = self.A_rate
        A_elevator_ang = self.A_elevator_ang
        state = self.parallel_state
        attack_ang, rate, elevator_ang = (
            self.parallel_state[:, 0],
            self.parallel_state[:, 1],
            self.parallel_state[:, 2],
        )
        # the elevator actuator voltage
        elevator_vol = action[:, 0]
        # wind gusts on angle of attack
        wind_attack_angle = action[:, 1]

        deri_attack_ang = torch.mm(state, A_attack_ang).squeeze(-1) + wind_attack_angle
        deri_rate = torch.mm(state, A_rate).squeeze(-1)
        deri_elevator_ang = torch.mm(state, A_elevator_ang).squeeze(-1) + elevator_vol

        delta_state = torch.stack(
            [deri_attack_ang, deri_rate, deri_elevator_ang], dim=-1
        )
        self.parallel_state = self.parallel_state + delta_state * dt

        reward = (
            self.Q[0][0] * attack_ang ** 2
            + self.Q[1][1] * rate ** 2
            + self.Q[2][2] * elevator_ang ** 2
            + self.R[0][0] * (elevator_vol ** 2).squeeze(-1)
            - self.gamma_atte ** 2 * (wind_attack_angle ** 2).squeeze(-1)
        )

        # define the ending condation here the format is just like isdone = l(next_state)
        done = (
            torch.where(
                abs(self.parallel_state[:, 0]) > self.attack_ang_threshold,
                self.ones_,
                self.zeros_,
            ).bool()
            | torch.where(
                abs(self.parallel_state[:, 1]) > self.rate_threshold,
                self.ones_,
                self.zeros_,
            ).bool()
            | torch.where(
                abs(self.parallel_state[:, 2]) > self.elevator_ang_threshold,
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
        A_attack_ang = self.A_attack_ang
        A_rate = self.A_rate
        A_elevator_ang = self.A_elevator_ang
        attack_ang, rate, elevator_ang = state[:, 0], state[:, 1], state[:, 2]
        # the elevator actuator voltage
        elevator_vol = action[:, 0]
        if self.is_adversary:
            # wind gusts on angle of attack
            wind_attack_angle = action[:, 1]
        else:
            wind_attack_angle = torch.zeros_like(elevator_vol)

        deri_attack_ang = torch.mm(state, A_attack_ang).squeeze(-1) + wind_attack_angle
        deri_rate = torch.mm(state, A_rate).squeeze(-1)
        deri_elevator_ang = torch.mm(state, A_elevator_ang).squeeze(-1) + elevator_vol

        delta_state = torch.stack(
            [deri_attack_ang, deri_rate, deri_elevator_ang], dim=-1
        )
        state_next = state + delta_state * dt
        cost = (
            self.Q[0][0] * attack_ang ** 2
            + self.Q[1][1] * rate ** 2
            + self.Q[2][2] * elevator_ang ** 2
            + self.R[0][0] * (elevator_vol ** 2).squeeze(-1)
            - self.gamma_atte ** 2 * (wind_attack_angle ** 2).squeeze(-1)
        )
        reward = -cost
        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = state[:, 0].new_zeros(size=[state.size()[0]], dtype=torch.bool)

        return state_next, reward, isdone, {"delta_state": delta_state}

    def f_x(self, state):
        batch_size = state.size()[0]

        if batch_size > 1:
            fx = torch.mm(state, self.A.t())
        else:
            fx = torch.mm(self.A, state.t())

        return fx

    def g_x(self, state, batch_size=1):

        if batch_size > 1:
            gx = torch.zeros((batch_size, self.state_dim, self.action_dim))
            for i in range(batch_size):
                gx[i, :, :] = self.B
        else:
            gx = self.B

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
            gx = self.B
            act = -0.5 * torch.mm(self.R.inverse(), torch.mm(gx.t(), delta_value.t()))

        return act.detach()

    def k_x(self, state, batch_size=1):

        if batch_size > 1:
            kx = torch.zeros((batch_size, self.state_dim, self.adversary_dim))
            for i in range(batch_size):
                kx[i, :, :] = self.D
        else:
            kx = self.D

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
            kx = self.D
            adv = 0.5 / (self.gamma_atte ** 2) * torch.mm(kx.t(), delta_value.t())

        return adv.detach()
