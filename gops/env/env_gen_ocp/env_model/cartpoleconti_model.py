from typing import Optional, Union

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.robot.cartpole_dynamics_model import CpDynMdl


class CartpolecontiMdl(EnvModel):
    dt: Optional[float] = 0.02
    action_dim: int = 1
    obs_dim: int = 4
    robot_model: CpDynMdl

    def __init__(self, device: Union[torch.device, str, None] = None):
        super().__init__(
            action_lower_bound=[-1.0],
            action_upper_bound=[1.0],
            device=device,
        )
        self.robot_model = CpDynMdl(device=device)

    def get_obs(self, state: State) -> torch.Tensor:
        return state.robot_state
    
    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        return 1 - self.get_terminated(state).float()
    
    def get_terminated(self, state: State) -> torch.Tensor:
        ref_x, ref_theta = (
            state.context_state.reference[:, 0],
            state.context_state.reference[:, 1],
        )
        x, theta= (
            state.robot_state[:, 0],
            state.robot_state[:, 2],
        )
        return (
            (torch.abs(x - ref_x) > self.robot_model.x_threshold)
            | (torch.abs(theta - ref_theta) > self.robot_model.theta_threshold_radians)
        )
    

def env_model_creator(**kwargs) -> CartpolecontiMdl:
    return CartpolecontiMdl()