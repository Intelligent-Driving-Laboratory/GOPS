#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Inverted double pendulum, model type
#  Update: 2022-12-05, Yuhang Zhang: create file

from typing import Tuple, Union
import torch
import numpy as np
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict


class Dynamics(object):
    def __init__(self):
        self.mass_cart = 9.42477796
        self.mass_rod1 = 4.1033127
        self.mass_rod2 = 4.1033127
        self.l_rod1 = 0.6
        self.l_rod2 = 0.6
        self.g = 9.81
        self.damping_cart = 0.0
        self.damping_rod1 = 0.0
        self.damping_rod2 = 0.0

    def f_xu(self, states, actions, tau):
        m, m1, m2 = self.mass_cart, self.mass_rod1, self.mass_rod2

        l1, l2 = self.l_rod1, self.l_rod2

        d1, d2, d3 = self.damping_cart, self.damping_rod1, self.damping_rod2

        g = self.g

        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )

        u = actions[:, 0]

        ones = torch.ones_like(p, dtype=torch.float32)
        M = torch.stack(
            [
                (m + m1 + m2) * ones,
                l1 * (0.5 * m1 + m2) * torch.cos(theta1),
                0.5 * m2 * l2 * torch.cos(theta2),
                l1 * (0.5 * m1 + m2) * torch.cos(theta1),
                l1 * l1 * (0.3333 * m1 + m2) * ones,
                0.5 * l1 * l2 * m2 * torch.cos(theta1 - theta2),
                0.5 * l2 * m2 * torch.cos(theta2),
                0.5 * l1 * l2 * m2 * torch.cos(theta1 - theta2),
                0.3333 * l2 * l2 * m2 * ones,
            ],
            dim=1,
        ).reshape(-1, 3, 3)

        f = torch.stack(
            [
                l1 * (0.5 * m1 + m2) * torch.square(theta1dot) * torch.sin(theta1)
                + 0.5 * m2 * l2 * torch.square(theta2dot) * torch.sin(theta2)
                - d1 * pdot
                + u,
                -0.5
                * l1
                * l2
                * m2
                * torch.square(theta2dot)
                * torch.sin(theta1 - theta2)
                + g * (0.5 * m1 + m2) * l1 * torch.sin(theta1)
                - d2 * theta1dot,
                0.5
                * l1
                * l2
                * m2
                * torch.square(theta1dot)
                * torch.sin(theta1 - theta2)
                + g * 0.5 * l2 * m2 * torch.sin(theta2),
            ],
            dim=1,
        ).reshape(-1, 3, 1)

        M_inv = torch.linalg.inv(M)
        tmp = torch.matmul(M_inv, f).squeeze(-1)

        deriv = torch.cat([states[:, 3:], tmp], dim=-1)
        next_states = states + tau * deriv
        next_p, next_theta1, next_theta2, next_pdot, next_theta1dot, next_theta2dot = (
            next_states[:, 0],
            next_states[:, 1],
            next_states[:, 2],
            next_states[:, 3],
            next_states[:, 4],
            next_states[:, 5],
        )

        next_p = next_p.reshape(-1, 1)
        next_theta1 = next_theta1.reshape(-1, 1)
        next_theta2 = next_theta2.reshape(-1, 1)
        next_pdot = next_pdot.reshape(-1, 1)
        next_theta1dot = next_theta1dot.reshape(-1, 1)
        next_theta2dot = next_theta2dot.reshape(-1, 1)
        next_states = torch.cat(
            [
                next_p,
                next_theta1,
                next_theta2,
                next_pdot,
                next_theta1dot,
                next_theta2dot,
            ],
            dim=1,
        )
        # print(next_states.shape, "-------------")
        return next_states

    def compute_rewards(self, states, actions):  # obses and actions are tensors
        actions = actions.squeeze(-1)
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )
        tip_x = p + self.l_rod1 * torch.sin(theta1) + self.l_rod2 * torch.sin(theta2)
        tip_y = self.l_rod1 * torch.cos(theta1) + self.l_rod2 * torch.cos(theta2)

        dist_penalty = (
            0 * torch.square(p) + 5 * torch.square(theta1) + 10 * torch.square(theta2)
        )
        v0, v1, v2 = pdot, theta1dot, theta2dot
        vel_penalty = (
            0.5 * torch.square(v0) + 0.5 * torch.square(v1) + 1 * torch.square(v2)
        )
        act_penalty = 1 * torch.square(actions)
        rewards = 10 - dist_penalty - vel_penalty - act_penalty

        return rewards

    def get_done(self, states):
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )
        point0x, point0y = p, 0
        point1x, point1y = (
            point0x + self.l_rod1 * torch.sin(theta1),
            point0y + self.l_rod1 * torch.cos(theta1),
        )
        point2x, point2y = (
            point1x + self.l_rod2 * torch.sin(theta2),
            point1y + self.l_rod2 * torch.cos(theta2),
        )

        d1 = point2y <= 1.0
        d2 = torch.abs(point0x) >= 15
        return torch.logical_or(d1, d2)  # point2y <= 1.0


class PythInvertedpendulum(PythBaseModel):
    def __init__(self, device: Union[torch.device, str, None] = None):
        obs_dim = 6
        action_dim = 1
        dt = 0.01
        self.discrete_num = 5
        lb_state = [-np.inf] * obs_dim
        hb_state = [np.inf] * obs_dim
        lb_action = [-1.0]
        hb_action = [1.0]
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            dt=dt,
            obs_lower_bound=lb_state,
            obs_upper_bound=hb_state,
            action_lower_bound=lb_action,
            action_upper_bound=hb_action,
            device=device,
        )
        # define your custom parameters here

        self.dynamics = Dynamics()

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        next_obs = obs
        for _ in range(self.discrete_num):
            next_obs = self.dynamics.f_xu(
                obs, 500 * action, self.dt / self.discrete_num
            )
            obs = next_obs
        reward = self.dynamics.compute_rewards(next_obs, action).reshape(-1)
        # done = torch.full([obs.size()[0]], False, dtype=torch.bool, device=self.device)
        done = self.dynamics.get_done(next_obs).reshape(-1)
        info = {"constraint": None}
        return next_obs, reward, done, info


def env_model_creator(**kwargs):
    """
    make env model `pyth_invertedpendulum`
    """
    return PythInvertedpendulum(kwargs.get("device", None))
