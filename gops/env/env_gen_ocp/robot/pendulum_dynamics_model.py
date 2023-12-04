from typing import Optional, Sequence

import torch
from torch.types import Device
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.robot.pendulum_dynamics import PendulumParam


class PendulumDynamicsModel(RobotModel):
    dt: float = 0.05
    robot_state_dim: int = 2

    def __init__(
        self, 
        robot_state_lower_bound: Optional[Sequence] = None, 
        robot_state_upper_bound: Optional[Sequence] = None, 
        device: Device = None,
    ):
        super().__init__(
            robot_state_lower_bound=robot_state_lower_bound,
            robot_state_upper_bound=robot_state_upper_bound,
            device=device,
        )
        self.param = PendulumParam()

    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        th, thdot = state[:, 0], state[:, 1]

        g = self.param.g
        m = self.param.m
        l = self.param.l
        dt = self.dt

        u = action[:, 0]

        newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = torch.clamp(newthdot, -self.param.max_speed, self.param.max_speed)
        newth = th + newthdot * dt

        next_state = torch.stack((newth, newthdot), dim=-1)
        return next_state
