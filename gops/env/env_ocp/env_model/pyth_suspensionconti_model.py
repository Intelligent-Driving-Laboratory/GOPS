#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Suspension Model
#  Update Date: 2022-08-12, Jie Li: create environment
#  Update Date: 2022-10-24, Yujie Yang: add wrapper

from typing import Tuple, Union

import torch
import numpy as np

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict

pi = torch.tensor(np.pi, dtype=torch.float32)


class PythSuspensioncontiModel(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None, **kwargs):
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs["is_adversary"]
        self.sample_batch_size = kwargs["reset_batch_size"]
        self.state_dim = 4
        self.action_dim = 1
        self.adversary_dim = 1
        self.dt = 1 / 500

        # define your custom parameters here
        # the mass of the car body [kg]
        self.M_b = 300
        # the mass of the wheel [kg]
        self.M_us = 60
        # the tyre stiffness [N/m]
        self.K_t = 190000
        # the linear suspension stiffness [N/m]
        self.K_a = 16000
        # the nonlinear suspension stiffness [N/m]
        self.K_n = self.K_a / 10
        # the damping rate of the suspension [N / (m/s)]
        self.C_a = 1000
        self.control_gain = 1e3

        # utility information
        self.state_weight = kwargs["state_weight"]
        self.control_weight = kwargs["control_weight"]
        self.Q = torch.zeros((self.state_dim, self.state_dim))
        self.Q[0][0] = self.state_weight[0]
        self.Q[1][1] = self.state_weight[1]
        self.Q[2][2] = self.state_weight[2]
        self.Q[3][3] = self.state_weight[3]
        self.R = torch.zeros((self.action_dim, self.action_dim))
        self.R[0][0] = self.control_weight[0]
        self.gamma = 1
        self.gamma_atte = kwargs["gamma_atte"]

        # state & action space
        self.fixed_initial_state = kwargs["fixed_initial_state"]
        self.initial_state_range = kwargs["initial_state_range"]
        self.pos_body_initial = self.initial_state_range[0]
        self.vel_body_initial = self.initial_state_range[1]
        self.pos_wheel_initial = self.initial_state_range[2]
        self.vel_wheel_initial = self.initial_state_range[3]
        self.state_threshold = kwargs["state_threshold"]
        self.pos_body_threshold = float(self.state_threshold[0])
        self.vel_body_threshold = float(self.state_threshold[1])
        self.pos_wheel_threshold = float(self.state_threshold[2])
        self.vel_wheel_threshold = float(self.state_threshold[3])
        self.min_action = [-1.2]
        self.max_action = [1.2]
        self.min_adv_action = [-2.0 / self.gamma_atte]
        self.max_adv_action = [2.0 / self.gamma_atte]

        self.lb_state = torch.tensor(
            [
                -self.pos_body_threshold,
                -self.vel_body_threshold,
                -self.pos_wheel_threshold,
                -self.vel_wheel_threshold,
            ],
            dtype=torch.float32,
        )
        self.hb_state = torch.tensor(
            [
                self.pos_body_threshold,
                self.vel_body_threshold,
                self.pos_wheel_threshold,
                self.vel_wheel_threshold,
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

        pos_body = np.random.uniform(
            -self.pos_body_initial, self.pos_body_initial, [self.sample_batch_size, 1]
        )
        vel_body = np.random.uniform(
            -self.vel_body_initial, self.vel_body_initial, [self.sample_batch_size, 1]
        )
        pos_wheel = np.random.uniform(
            -self.pos_wheel_initial, self.pos_wheel_initial, [self.sample_batch_size, 1]
        )
        vel_wheel = np.random.uniform(
            -self.vel_wheel_initial, self.vel_wheel_initial, [self.sample_batch_size, 1]
        )

        state = np.concatenate([pos_body, vel_body, pos_wheel, vel_wheel], axis=1)

        return torch.from_numpy(state).float()

    def step(self, action: torch.Tensor):
        dt = self.dt
        M_b = self.M_b
        M_us = self.M_us
        K_t = self.K_t
        K_a = self.K_a
        K_n = self.K_n
        C_a = self.C_a
        control_gain = self.control_gain
        pos_body, vel_body = self.parallel_state[:, 0], self.parallel_state[:, 1]
        pos_wheel, vel_wheel = self.parallel_state[:, 2], self.parallel_state[:, 3]
        # the control force of the hydraulic actuator [kN]
        force = action[:, 0]
        # the road disturbance [m]
        pos_road = action[:, 1]

        deri_pos_body = vel_body
        deri_vel_body = (
            -(
                K_a * (pos_body - pos_wheel)
                + K_n * torch.pow(pos_body - pos_wheel, 3)
                + C_a * (vel_body - vel_wheel)
                - control_gain * force
            )
            / M_b
        )
        deri_pos_wheel = vel_wheel
        deri_vel_wheel = (
            K_a * (pos_body - pos_wheel)
            + K_n * torch.pow(pos_body - pos_wheel, 3)
            + C_a * (vel_body - vel_wheel)
            - K_t * (pos_wheel - pos_road)
            - control_gain * force
        ) / M_us

        delta_state = torch.stack(
            [deri_pos_body, deri_vel_body, deri_pos_wheel, deri_vel_wheel], dim=-1
        )
        self.parallel_state = self.parallel_state + delta_state * dt

        reward = (
            self.Q[0][0] * pos_body ** 2
            + self.Q[1][1] * vel_body ** 2
            + self.Q[2][2] * pos_wheel ** 2
            + self.Q[3][3] * vel_wheel ** 2
            + self.R[0][0] * (force ** 2).squeeze(-1)
            - self.gamma_atte ** 2 * (pos_road ** 2).squeeze(-1)
        )

        # define the ending condation here the format is just like isdone = l(next_state)
        done = (
            torch.where(
                abs(self.parallel_state[:, 0]) > self.pos_body_threshold,
                self.ones_,
                self.zeros_,
            ).bool()
            | torch.where(
                abs(self.parallel_state[:, 1]) > self.vel_body_threshold,
                self.ones_,
                self.zeros_,
            ).bool()
            | torch.where(
                abs(self.parallel_state[:, 2]) > self.pos_wheel_threshold,
                self.ones_,
                self.zeros_,
            ).bool()
            | torch.where(
                abs(self.parallel_state[:, 3]) > self.vel_wheel_threshold,
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
        M_b = self.M_b
        M_us = self.M_us
        K_t = self.K_t
        K_a = self.K_a
        K_n = self.K_n
        C_a = self.C_a
        control_gain = self.control_gain
        pos_body, vel_body, pos_wheel, vel_wheel = (
            state[:, 0],
            state[:, 1],
            state[:, 2],
            state[:, 3],
        )
        # the control force of the hydraulic actuator [kN]
        force = action[:, 0]
        if self.is_adversary:
            # the road disturbance [m]
            pos_road = action[:, 1]
        else:
            pos_road = torch.zeros_like(force)

        deri_pos_body = vel_body
        deri_vel_body = (
            -(
                K_a * (pos_body - pos_wheel)
                + K_n * torch.pow(pos_body - pos_wheel, 3)
                + C_a * (vel_body - vel_wheel)
                - control_gain * force
            )
            / M_b
        )
        deri_pos_wheel = vel_wheel
        deri_vel_wheel = (
            K_a * (pos_body - pos_wheel)
            + K_n * torch.pow(pos_body - pos_wheel, 3)
            + C_a * (vel_body - vel_wheel)
            - K_t * (pos_wheel - pos_road)
            - control_gain * force
        ) / M_us

        delta_state = torch.stack(
            [deri_pos_body, deri_vel_body, deri_pos_wheel, deri_vel_wheel], dim=-1
        )
        state_next = state + delta_state * dt
        cost = (
            self.Q[0][0] * pos_body ** 2
            + self.Q[1][1] * vel_body ** 2
            + self.Q[2][2] * pos_wheel ** 2
            + self.Q[3][3] * vel_wheel ** 2
            + self.R[0][0] * (force ** 2).squeeze(-1)
            - self.gamma_atte ** 2 * (pos_road ** 2).squeeze(-1)
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

        M_b = self.M_b
        M_us = self.M_us
        K_t = self.K_t
        K_a = self.K_a
        K_n = self.K_n
        C_a = self.C_a

        if batch_size > 1:
            fx = torch.zeros((batch_size, self.state_dim))
            fx[:, 0] = state[:, 1]
            fx[:, 1] = (
                -(
                    K_a * (state[:, 0] - state[:, 2])
                    + K_n * torch.pow(state[:, 0] - state[:, 2], 3)
                    + C_a * (state[:, 1] - state[:, 3])
                )
                / M_b
            )
            fx[:, 2] = state[:, 3]
            fx[:, 3] = (
                K_a * (state[:, 0] - state[:, 2])
                + K_n * torch.pow(state[:, 0] - state[:, 2], 3)
                + C_a * (state[:, 1] - state[:, 3])
                - K_t * state[:, 2]
            ) / M_us
        else:
            fx = torch.zeros((self.state_dim, 1))
            fx[0, 0] = state[0, 1]
            fx[1, 0] = (
                -(
                    K_a * (state[0, 0] - state[0, 2])
                    + K_n * torch.pow(state[0, 0] - state[0, 2], 3)
                    + C_a * (state[0, 1] - state[0, 3])
                )
                / M_b
            )
            fx[2, 0] = state[0, 3]
            fx[3, 0] = (
                K_a * (state[0, 0] - state[0, 2])
                + K_n * torch.pow(state[0, 0] - state[0, 2], 3)
                + C_a * (state[0, 1] - state[0, 3])
                - K_t * state[0, 2]
            ) / M_us

        return fx

    def g_x(self, state, batch_size):

        if batch_size > 1:
            gx = torch.zeros((batch_size, self.state_dim, self.action_dim))
            gx[:, 0, 0] = torch.zeros((batch_size,))
            gx[:, 1, 0] = self.control_gain / self.M_b * torch.ones((batch_size,))
            gx[:, 2, 0] = torch.zeros((batch_size,))
            gx[:, 3, 0] = -self.control_gain / self.M_us * torch.ones((batch_size,))
        else:
            gx = torch.zeros((self.state_dim, self.action_dim))
            gx[0, 0] = 0
            gx[1, 0] = self.control_gain / self.M_b
            gx[2, 0] = 0
            gx[3, 0] = -self.control_gain / self.M_us

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
            kx[:, 1, 0] = torch.zeros((batch_size,))
            kx[:, 2, 0] = torch.zeros((batch_size,))
            kx[:, 3, 0] = self.K_t / self.M_us * torch.ones((batch_size,))
        else:
            kx = torch.zeros((self.state_dim, self.adversary_dim))
            kx[0, 0] = 0
            kx[1, 0] = 0
            kx[2, 0] = 0
            kx[3, 0] = self.K_t / self.M_us

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
