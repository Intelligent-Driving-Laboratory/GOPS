from typing import Optional, Tuple, Union

import torch

from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.robot.veh3dof_model import Veh3DoFModel
from gops.env.env_gen_ocp.robot.veh3dof import angle_normalize


class Veh3DoFTrackingModel(EnvModel):
    dt: Optional[float] = 0.1
    action_dim: int = 2
    robot_model: Veh3DoFModel

    def __init__(
        self,
        pre_horizon: int = 10,
        max_acc: float = 3.0,
        max_steer: float = torch.pi / 6,
        device: Union[torch.device, str, None] = None,
        **kwargs,
    ):
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.obs_dim = ego_obs_dim + ref_obs_dim * pre_horizon
        super().__init__(
            obs_lower_bound=None,
            obs_upper_bound=None,
            action_lower_bound=[-max_steer, -max_acc],
            action_upper_bound=[max_steer, max_acc],
            device=device,
        )
        self.robot_model = Veh3DoFModel()
        self.pre_horizon = pre_horizon

    def get_obs(self, state: State) -> torch.Tensor:
        t = state.context_state.t
        current_reference = state.context_state.reference[:, t:t + self.pre_horizon + 1]
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                state.robot_state[:, 0],
                state.robot_state[:, 1],
                state.robot_state[:, 2],
                current_reference[..., 0],
                current_reference[..., 1],
                current_reference[..., 2],
            )
        ref_u_tf = current_reference[..., 3] - state.robot_state[:, 3].unsqueeze(1)
        ego_obs = torch.concat((torch.stack(
            (ref_x_tf[:, 0], ref_y_tf[:, 0], ref_phi_tf[:, 0], ref_u_tf[:, 0]), 
            dim=1
            ), state.robot_state[:, 4:]), dim=1
        )
        ref_obs = torch.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 2)[:, 1:] \
            .reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs), 1)

    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        ego_obs = state.robot_state
        x, y, phi, u, w = ego_obs[:, 0], ego_obs[:, 1], ego_obs[:, 2], ego_obs[:, 3], ego_obs[:, 5]
        ref_obs = state.context_state.index_by_t().reference
        ref_x, ref_y, ref_phi, ref_u = ref_obs[:, 0], ref_obs[:, 1], ref_obs[:, 2], ref_obs[:, 3]
        steer, a_x = action[:, 0], action[:, 1]
        return -(
            0.04 * (x - ref_x) ** 2
            + 0.04 * (y - ref_y) ** 2
            + 0.02 * angle_normalize(phi - ref_phi) ** 2
            + 0.02 * (u - ref_u) ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
            + 0.01 * a_x ** 2
        )

    def get_terminated(self, state: State) -> torch.bool:
        ego_obs = state.robot_state
        x, y, phi = ego_obs[:, 0], ego_obs[:, 1], ego_obs[:, 2]
        ref_obs = state.context_state.index_by_t().reference
        ref_x, ref_y, ref_phi = ref_obs[:, 0], ref_obs[:, 1], ref_obs[:, 2]
        done = (
            (torch.abs(x - ref_x) > 5)
            | (torch.abs(y - ref_y) > 2)
            | (torch.abs(angle_normalize(phi - ref_phi)) > torch.pi)
        )
        return done


def ego_vehicle_coordinate_transform(
    ego_x: torch.Tensor,
    ego_y: torch.Tensor,
    ego_phi: torch.Tensor,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ref_phi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_x, ego_y, ego_phi = ego_x.unsqueeze(1), ego_y.unsqueeze(1), ego_phi.unsqueeze(1)
    cos_tf = torch.cos(-ego_phi)
    sin_tf = torch.sin(-ego_phi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_phi_tf = angle_normalize(ref_phi - ego_phi)
    return ref_x_tf, ref_y_tf, ref_phi_tf


def env_model_creator(**kwargs) -> Veh3DoFTrackingModel:
    return Veh3DoFTrackingModel(**kwargs)

