#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Suspension Environment
#

import warnings
import torch
import numpy as np

pi = torch.tensor(np.pi, dtype=torch.float32)


class PythSuspensionModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs['is_adversary']
        self.sample_batch_size = kwargs['sample_batch_size']
        self.state_dim = 4
        self.action_dim = 1
        self.adversary_dim = 1
        self.dt = 1 / 500

        # define your custom parameters here
        self.M_b = 300  # the mass of the car body(kg)
        self.M_us = 60  # the mass of the wheel(kg)
        self.K_t = 190000  # the tyre stiffness(N/m)
        self.K_a = 16000  # the linear suspension stiffness(N/m)
        self.K_n = self.K_a / 10  # the nonlinear suspension stiffness(N/m)
        self.C_a = 1000  # the damping rate of the suspension(N/(m/s))
        self.control_gain = 1e3

        # utility information
        self.Q = torch.eye(self.state_dim)
        self.Q[0, 0] = 1000.
        self.Q[1, 1] = 3.
        self.Q[2, 2] = 100.
        self.Q[3, 3] = 0.1
        self.R = torch.eye(self.action_dim)
        self.gamma = 1
        self.gamma_atte = 30.0

        # state & action space
        self.pos_body_initial = 0.05
        self.vel_body_initial = 0.5
        self.pos_wheel_initial = 0.05
        self.vel_wheel_initial = 1.0
        self.pos_body_threshold = 0.08
        self.vel_body_threshold = 0.6
        self.pos_wheel_threshold = 0.1
        self.vel_wheel_threshold = 1.6
        self.min_action = [-1.2]
        self.max_action = [1.2]
        self.min_adv_action = [-2.0 / self.gamma_atte]
        self.max_adv_action = [2.0 / self.gamma_atte]

        self.lb_state = torch.tensor([-self.pos_body_threshold, -self.vel_body_threshold,
                                      -self.pos_wheel_threshold, -self.vel_wheel_threshold], dtype=torch.float32)
        self.hb_state = torch.tensor([self.pos_body_threshold, self.vel_body_threshold,
                                      self.pos_wheel_threshold, self.vel_wheel_threshold], dtype=torch.float32)
        self.lb_action = torch.tensor(self.min_action + self.min_adv_action, dtype=torch.float32)  # action & adversary
        self.hb_action = torch.tensor(self.max_action + self.max_adv_action, dtype=torch.float32)

        self.ones_ = torch.ones(self.sample_batch_size)
        self.zeros_ = torch.zeros(self.sample_batch_size)

        # parallel sample
        self.parallel_state = None
        self.max_step_per_episode = self.max_step()
        self.step_per_episode = self.initial_step()

    def max_step(self):
        return torch.from_numpy(np.floor(np.random.uniform(200, 500, [self.sample_batch_size])))

    def initial_step(self):
        return torch.zeros(self.sample_batch_size)

    def reset(self):

        pos_body = np.random.uniform(-self.pos_body_initial, self.pos_body_initial, [self.sample_batch_size, 1])
        vel_body = np.random.uniform(-self.vel_body_initial, self.vel_body_initial, [self.sample_batch_size, 1])
        pos_wheel = np.random.uniform(-self.pos_wheel_initial, self.pos_wheel_initial, [self.sample_batch_size, 1])
        vel_wheel = np.random.uniform(-self.vel_wheel_initial, self.vel_wheel_initial, [self.sample_batch_size, 1])

        state = np.concatenate([pos_body, vel_body, pos_wheel, vel_wheel], axis=1)  # concatenate column

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
        force = action[:, 0]  # the control force of the hydraulic actuator [kN]
        pos_road = action[:, 1]  # the road disturbance

        deri_pos_body = vel_body
        deri_vel_body = - (K_a * (pos_body - pos_wheel) + K_n * torch.pow(pos_body - pos_wheel, 3) +
                           C_a * (vel_body - vel_wheel) - control_gain * force) / M_b
        deri_pos_wheel = vel_wheel
        deri_vel_wheel = (K_a * (pos_body - pos_wheel) + K_n * torch.pow(pos_body - pos_wheel, 3) +
                          C_a * (vel_body - vel_wheel) - K_t * (pos_wheel - pos_road) - control_gain * force) / M_us

        delta_state = torch.stack([deri_pos_body, deri_vel_body, deri_pos_wheel, deri_vel_wheel], dim=-1)
        self.parallel_state = self.parallel_state + delta_state * dt

        reward = (self.Q[0][0] * pos_body ** 2 + self.Q[1][1] * vel_body ** 2
                  + self.Q[2][2] * pos_wheel ** 2 + self.Q[3][3] * vel_wheel ** 2
                  + self.R[0][0] * (force ** 2).squeeze(-1) - self.gamma_atte ** 2 * (pos_road ** 2).squeeze(-1))

        # define the ending condation here the format is just like isdone = l(next_state)
        done = (torch.where(abs(self.parallel_state[:, 0]) > self.pos_body_threshold, self.ones_, self.zeros_).bool()
                | torch.where(abs(self.parallel_state[:, 1]) > self.vel_body_threshold, self.ones_, self.zeros_).bool()
                | torch.where(abs(self.parallel_state[:, 2]) > self.pos_wheel_threshold, self.ones_, self.zeros_).bool()
                | torch.where(abs(self.parallel_state[:, 3]) > self.vel_wheel_threshold, self.ones_, self.zeros_).bool())

        self.step_per_episode += 1
        info = {'TimeLimit.truncated': torch.where(self.step_per_episode > self.max_step_per_episode,
                                                   self.ones_, self.zeros_).bool()}

        return self.parallel_state, reward, done, info

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
            # warnings.warn(warning_msg)
            action = clip_by_tensor(action, self.lb_action, self.hb_action)

        warning_msg = "state out of state space!"
        if not ((state <= self.hb_state).all() and (state >= self.lb_state).all()):
            # warnings.warn(warning_msg)
            state = clip_by_tensor(state, self.lb_state, self.hb_state)

        dt = self.dt
        M_b = self.M_b
        M_us = self.M_us
        K_t = self.K_t
        K_a = self.K_a
        K_n = self.K_n
        C_a = self.C_a
        control_gain = self.control_gain
        pos_body, vel_body, pos_wheel, vel_wheel = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        force = action[:, 0]     # the control force of the hydraulic actuator [kN]
        pos_road = action[:, 1]  # the road disturbance

        deri_pos_body = vel_body
        deri_vel_body = - (K_a * (pos_body - pos_wheel) + K_n * torch.pow(pos_body - pos_wheel, 3) +
                           C_a * (vel_body - vel_wheel) - control_gain * force) / M_b
        deri_pos_wheel = vel_wheel
        deri_vel_wheel = (K_a * (pos_body - pos_wheel) + K_n * torch.pow(pos_body - pos_wheel, 3) +
                          C_a * (vel_body - vel_wheel) - K_t * (pos_wheel - pos_road) - control_gain * force) / M_us

        delta_state = torch.stack([deri_pos_body, deri_vel_body, deri_pos_wheel, deri_vel_wheel], dim=-1)
        state_next = state + delta_state * dt
        reward = (self.Q[0][0] * pos_body ** 2 + self.Q[1][1] * vel_body ** 2
                  + self.Q[2][2] * pos_wheel ** 2 + self.Q[3][3] * vel_wheel ** 2
                  + self.R[0][0] * (force ** 2).squeeze(-1) - self.gamma_atte ** 2 * (pos_road ** 2).squeeze(-1))
        ############################################################################################

        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = state[:, 0].new_zeros(size=[state.size()[0]], dtype=torch.bool)

        ############################################################################################
        # beyond_done = beyond_done.bool()
        # mask = isdone * beyond_done
        # mask = torch.unsqueeze(mask, -1)
        # state_next = ~mask * state_next + mask * state
        return delta_state, reward, isdone

    def g_x(self, state, batch_size):

        gx = torch.zeros((batch_size, self.action_dim, self.state_dim))
        gx[:, 0, 0] = torch.zeros((batch_size, ))
        gx[:, 0, 1] = self.control_gain / self.M_b * torch.ones((batch_size, ))
        gx[:, 0, 2] = torch.zeros((batch_size, ))
        gx[:, 0, 3] = - self.control_gain / self.M_us * torch.ones((batch_size, ))

        return gx

    def best_act(self, state, delta_value):
        batch_size = state.size()[0]

        if batch_size > 1:
            gx = self.g_x(state, batch_size)  # [64, 1, 4]
            delta_value = delta_value[:, :, np.newaxis]  # [64, 4, 1]
            act = - 0.5 * torch.matmul(self.R.inverse(), torch.bmm(gx, delta_value)).squeeze(-1)  # [64, 4]
        else:
            gx = torch.tensor([[0., self.control_gain / self.M_b, 0., -self.control_gain / self.M_us]])
            act = - 0.5 * torch.mm(self.R.inverse(), torch.mm(gx, delta_value.t()))

        return act.detach()

    def k_x(self, state, batch_size):

        kx = torch.zeros((batch_size, self.adversary_dim, self.state_dim))
        kx[:, 0, 0] = torch.zeros((batch_size, ))
        kx[:, 0, 1] = torch.zeros((batch_size, ))
        kx[:, 0, 2] = torch.zeros((batch_size, ))
        kx[:, 0, 3] = self.K_t / self.M_us * torch.ones((batch_size, ))

        return kx

    def worst_adv(self, state, delta_value):
        batch_size = state.size()[0]

        if batch_size > 1:
            kx = self.k_x(state, batch_size)  # [64, 1, 4]
            delta_value = delta_value[:, :, np.newaxis]  # [64, 4, 1]
            adv = 0.5 / (self.gamma_atte ** 2) * torch.bmm(kx, delta_value).squeeze(-1)  # [64, 4]
        else:
            kx = torch.tensor([[0., 0., 0., self.K_t / self.M_us]])
            adv = 0.5 / (self.gamma_atte ** 2) * torch.mm(kx, delta_value.t())

        return adv.detach()


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
