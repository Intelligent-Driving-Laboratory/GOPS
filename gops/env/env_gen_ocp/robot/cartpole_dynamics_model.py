from typing import Optional

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.robot.cartpole_dynamics import CartpoleParam


class CpDynMdl(RobotModel):
    dt: Optional[float] = 0.02
    robot_state_dim: int = 4
    theta_threshold_radians = 12 * 2 * torch.pi / 360
    x_threshold = 2.4

    def __init__(self, device: Optional[torch.device]):
        super().__init__(device=device)
        self.param = CartpoleParam()

    def get_next_state(
        self,
        robot_state: torch.Tensor,
        robot_action: torch.Tensor,
    ) -> torch.Tensor:
        gravity = self.param.gravity
        masspole = self.param.masspole
        total_mass = self.param.total_mass
        length = self.param.length
        polemass_length = self.param.polemass_length

        x, x_dot, theta, theta_dot = (
            robot_state[:, 0],
            robot_state[:, 1],
            robot_state[:, 2],
            robot_state[:, 3],
        )
        force = self.param.force_mag * robot_action[:, 0]

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (
            torch.squeeze(force) + polemass_length * theta_dot * theta_dot * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length
            * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
        )

        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        return torch.stack([x, x_dot, theta, theta_dot]).transpose(1, 0)