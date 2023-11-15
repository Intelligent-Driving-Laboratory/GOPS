from typing import Optional, Sequence

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.robot.veh3dof import angle_normalize, Veh3DoFParam


class Veh3DoFModel(RobotModel):
    dt: Optional[float] = 0.1
    robot_state_dim: int = 6

    def __init__(
        self, 
        robot_state_lower_bound: Optional[Sequence] = None, 
        robot_state_upper_bound: Optional[Sequence] = None, 
    ):
        super().__init__(
            robot_state_lower_bound=robot_state_lower_bound,
            robot_state_upper_bound=robot_state_upper_bound,
        )
        self.vehicle_params = Veh3DoFParam()

    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, y, phi, u, v, w = (
            state[:, 0],
            state[:, 1],
            state[:, 2],
            state[:, 3],
            state[:, 4],
            state[:, 5],
        )
        steer, a_x = action[:, 0], action[:, 1]
        k_f = self.vehicle_params.kf
        k_r = self.vehicle_params.kr
        l_f = self.vehicle_params.lf
        l_r = self.vehicle_params.lr
        m = self.vehicle_params.m
        I_z = self.vehicle_params.Iz
        next_state = [
            x + self.dt * (u * torch.cos(phi) - v * torch.sin(phi)),
            y + self.dt * (u * torch.sin(phi) + v * torch.cos(phi)),
            angle_normalize(phi + self.dt * w),
            u + self.dt * a_x,
            (
                m * v * u
                + self.dt * (l_f * k_f - l_r * k_r) * w
                - self.dt * k_f * steer * u
                - self.dt * m * torch.square(u) * w
            )
            / (m * u - self.dt * (k_f + k_r)),
            (
                I_z * w * u
                + self.dt * (l_f * k_f - l_r * k_r) * v
                - self.dt * l_f * k_f * steer * u
            )
            / (I_z * u - self.dt * (l_f ** 2 * k_f + l_r ** 2 * k_r)),
        ]
        return torch.stack(next_state, 1)