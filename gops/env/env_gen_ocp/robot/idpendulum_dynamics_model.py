from typing import Optional

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.robot.Idpendulum_dynamics import IdpendulumParam


class IdpDynMdl(RobotModel):
    dt: Optional[float] = 0.01
    robot_state_dim: int = 6

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.param = IdpendulumParam()
        self.discrete_num = 5

    def get_next_state(
        self, 
        robot_state: torch.Tensor, 
        action: torch.Tensor
        ) -> torch.Tensor:
        action = 500 * action
        for _ in range(self.discrete_num):
            robot_state = self._step(robot_state, action)
        return robot_state

    def _step(
        self, 
        robot_state: torch.Tensor, 
        robot_action: torch.Tensor
        ) -> torch.Tensor:
        state = robot_state
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            state[:, 0],
            state[:, 1],
            state[:, 2],
            state[:, 3],
            state[:, 4],
            state[:, 5],
        )
        u = robot_action[:, 0]

        m  = self.param.mass_cart
        m1 = self.param.mass_rod1
        m2 = self.param.mass_rod2
        l1 = self.param.l_rod1
        l2 = self.param.l_rod2
        g  = self.param.g
        d  = self.param.damping_cart
        d1 = self.param.damping_rod1
        d2 = self.param.damping_rod2
        tau = self.dt / self.discrete_num

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

        deriv = torch.cat([state[:, 3:], tmp], dim=-1)
        next_state = state + tau * deriv
        next_p, next_theta1, next_theta2, next_pdot, next_theta1dot, next_theta2dot = (
            next_state[:, 0],
            next_state[:, 1],
            next_state[:, 2],
            next_state[:, 3],
            next_state[:, 4],
            next_state[:, 5],
        )

        next_p = next_p.reshape(-1, 1)
        next_theta1 = next_theta1.reshape(-1, 1)
        next_theta2 = next_theta2.reshape(-1, 1)
        next_pdot = next_pdot.reshape(-1, 1)
        next_theta1dot = next_theta1dot.reshape(-1, 1)
        next_theta2dot = next_theta2dot.reshape(-1, 1)
        next_state = torch.cat(
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
        return next_state
        