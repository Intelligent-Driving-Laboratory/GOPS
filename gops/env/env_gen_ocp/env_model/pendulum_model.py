import numpy as np
import torch
from torch.types import Device
from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.robot.pendulum_dynamics_model import PendulumDynamicsModel


class PendulumModel(EnvModel):
    action_dim: int = 1
    obs_dim: int = 3
    robot_model: PendulumDynamicsModel

    def __init__(self, device: Device = None):
        self.robot_model = PendulumDynamicsModel(device=device)
        super().__init__(
            obs_lower_bound=[-1.0, -1.0, -self.robot_model.param.max_speed],
            obs_upper_bound=[1.0, 1.0, self.robot_model.param.max_speed],
            action_lower_bound=[-self.robot_model.param.max_torque],
            action_upper_bound=[self.robot_model.param.max_torque],
            device=device,
        )

    def get_obs(self, state: State) -> torch.Tensor:
        th, thdot = state.robot_state[:, 0], state.robot_state[:, 1]
        return torch.stack((torch.cos(th), torch.sin(th), thdot), dim=-1)
    
    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        th, thdot = state.robot_state[:, 0], state.robot_state[:, 1]

        th_targ, thdot_targ = state.context_state.reference[:, 0], state.context_state.reference[:, 1]

        u = torch.clamp(action, self.action_lower_bound, self.action_upper_bound)[:, 0]

        costs = (angle_normalize(th) - th_targ) ** 2 + \
            0.1 * (thdot - thdot_targ) ** 2 + 0.001 * (u ** 2)
        return -costs
    
    def get_terminated(self, state: State) -> torch.Tensor:
        return torch.zeros_like(state.robot_state[:, 0]).bool()


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
