from typing import Optional, Union

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.robot.veh2dof_model import Veh2DoFModel


class Veh2DoFTrackingModel(EnvModel):
    dt: Optional[float] = 0.1
    action_dim: int = 1
    robot_model: Veh2DoFModel

    def __init__(
        self,
        pre_horizon: int = 10,
        max_steer: float = torch.pi / 6,
        device: Union[torch.device, str, None] = None,
        **kwargs,
    ):
        ego_obs_dim = 4
        ref_obs_dim = 1
        self.obs_dim = ego_obs_dim + ref_obs_dim * pre_horizon
        super().__init__(
            obs_lower_bound=None,
            obs_upper_bound=None,
            action_lower_bound=[-max_steer],
            action_upper_bound=[max_steer],
            device=device,
        )
        self.robot_model = Veh2DoFModel()
        self.pre_horizon = pre_horizon

    def get_obs(self, state: State) -> torch.Tensor:
        t = state.context_state.t
        current_reference = state.context_state.reference[:, t:t + self.pre_horizon + 1]
        ego_obs = torch.concat((
            state.robot_state[:, :2] - current_reference[:, 0, 1:3],
            state.robot_state[:, 2:],
        ), dim=1)
        ref_obs = (
            state.robot_state[:, :1].unsqueeze(1) - current_reference[:, 1:, 1:2]
        ).reshape((-1, self.pre_horizon))
        return torch.concat((ego_obs, ref_obs), dim=1)

    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        y, phi, v, w = (
            state.robot_state[:, 0],
            state.robot_state[:, 1],
            state.robot_state[:, 2],
            state.robot_state[:, 3],
        )
        ref = state.context_state.index_by_t().reference
        ref_y, ref_phi = ref[:, 1], ref[:, 2]
        steer = action[:, 0]
        return -(
            0.04 * (y - ref_y) ** 2
            + 0.02 * (phi - ref_phi) ** 2
            + 0.01 * v ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
        )

    def get_terminated(self, state: State) -> torch.bool:
        y, phi = state.robot_state[:, 0], state.robot_state[:, 1]
        ref = state.context_state.index_by_t().reference
        ref_y, ref_phi = ref[:, 1], ref[:, 2]
        return (torch.abs(y - ref_y) > 2) | (torch.abs(phi - ref_phi) > torch.pi)


def env_model_creator(**kwargs) -> Veh2DoFTrackingModel:
    return Veh2DoFTrackingModel(**kwargs)

