from typing import Optional, Union

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.robot.idpendulum_dynamics_model import IdpDynMdl


class IdpendulumMdl(EnvModel):
    dt: Optional[float] = 0.01
    action_dim: int = 1
    obs_dim: int = 6
    robot_model: IdpDynMdl

    def __init__(self, device: Union[torch.device, str, None] = None):
        super().__init__(
            action_lower_bound=[-1.0],
            action_upper_bound=[1.0],
            device=device,
        )
        self.robot_model = IdpDynMdl(device=device)
    
    def get_obs(self, state: State) -> torch.Tensor:
        return state.robot_state
    
    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        action = action.squeeze(-1)
        ref_p, ref_theta1, ref_theta2 = (
            state.context_state.reference[:, 0],
            state.context_state.reference[:, 1],
            state.context_state.reference[:, 2],
        )
        p, theta1, theta2, pdot, theta1dot, theta2dot = (
            state.robot_state[:, 0] - ref_p,
            state.robot_state[:, 1] - ref_theta1,
            state.robot_state[:, 2] - ref_theta2,
            state.robot_state[:, 3],
            state.robot_state[:, 4],
            state.robot_state[:, 5],
        )
        dist_penalty = (
            0 * torch.square(p) + 5 * torch.square(theta1) + 10 * torch.square(theta2)
        )
        v0, v1, v2 = pdot, theta1dot, theta2dot
        vel_penalty = (
            0.5 * torch.square(v0) + 0.5 * torch.square(v1) + 1 * torch.square(v2)
        )
        act_penalty = 1 * torch.square(action)
        rewards = 10 - dist_penalty - vel_penalty - act_penalty
        return rewards

    def get_terminated(self, state: State) -> torch.Tensor:
        ref_p, ref_theta1, ref_theta2 = (
            state.context_state.reference[:, 0],
            state.context_state.reference[:, 1],
            state.context_state.reference[:, 2],
        )
        p, theta1, theta2 = (
            state.robot_state[:, 0] - ref_p,
            state.robot_state[:, 1] - ref_theta1,
            state.robot_state[:, 2] - ref_theta2,
        )
        l_rod1 = self.robot_model.param.l_rod1
        l_rod2 = self.robot_model.param.l_rod2
        point0x, point0y = p, 0
        point1x, point1y = (
            point0x + l_rod1 * torch.sin(theta1),
            point0y + l_rod1 * torch.cos(theta1),
        )
        point2x, point2y = (
            point1x + l_rod2 * torch.sin(theta2),
            point1y + l_rod2 * torch.cos(theta2),
        )

        d1 = point2y <= 1.0
        d2 = torch.abs(point0x) >= 15
        return torch.logical_or(d1, d2)


def env_model_creator(**kwargs) -> IdpendulumMdl:
    """
    make env model `idpendulum`
    """
    return IdpendulumMdl(kwargs.get("device", None))
