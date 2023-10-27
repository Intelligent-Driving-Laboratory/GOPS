from typing import Optional, Sequence

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.robot.veh3dof import angle_normalize


class VehDynMdl(RobotModel):
    def __init__(
        self, 
        dt: float = 0.1,
        robot_state_dim: int =  6, 
        robot_state_lower_bound: Optional[Sequence] = None, 
        robot_state_upper_bound: Optional[Sequence] = None, 
    ):
        self.robot_state_dim = robot_state_dim
        self.robot_state_lower_bound = torch.full((robot_state_dim,), float('-inf')) if robot_state_lower_bound is None else robot_state_lower_bound
        self.robot_state_upper_bound = torch.full((robot_state_dim,), float('inf')) if robot_state_upper_bound is None else robot_state_upper_bound
        
        self.dt = dt
        self.vehicle_params = dict(
            k_f=-128915.5,  # front wheel cornering stiffness [N/rad]
            k_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
            l_f=1.06,  # distance from CG to front axle [m]
            l_r=1.85,  # distance from CG to rear axle [m]
            m=1412.0,  # mass [kg]
            I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
            miu=1.0,  # tire-road friction coefficient
            g=9.81,  # acceleration of gravity [m/s^2]
        )
        l_f, l_r, mass, g = (
            self.vehicle_params["l_f"],
            self.vehicle_params["l_r"],
            self.vehicle_params["m"],
            self.vehicle_params["g"],
        )
        F_zf, F_zr = l_r * mass * g / (l_f + l_r), l_f * mass * g / (l_f + l_r)
        self.vehicle_params.update(dict(F_zf=F_zf, F_zr=F_zr))

    def get_next_state(self, robot_state: torch.Tensor, robot_action: torch.Tensor) -> torch.Tensor:
        x, y, phi, u, v, w = (
            robot_state[:, 0],
            robot_state[:, 1],
            robot_state[:, 2],
            robot_state[:, 3],
            robot_state[:, 4],
            robot_state[:, 5],
        )
        steer, a_x = robot_action[:, 0], robot_action[:, 1]
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
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