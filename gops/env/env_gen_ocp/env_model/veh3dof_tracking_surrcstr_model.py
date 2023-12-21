from typing import Optional, Union

import torch
import numpy as np

from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.robot.veh3dof_model import Veh3DoFModel
from gops.env.env_gen_ocp.robot.veh3dof import angle_normalize
from gops.env.env_gen_ocp.env_model.veh3dof_tracking_model import ego_vehicle_coordinate_transform


class Veh3DoFTrackingSurrCstrModel(EnvModel):
    dt: Optional[float] = 0.1
    action_dim: int = 2
    robot_model: Veh3DoFModel

    def __init__(
        self,
        pre_horizon: int = 10,
        max_steer: float = torch.pi / 6,
        device: Union[torch.device, str, None] = None,
        veh_length: float = 4.8,
        veh_width: float = 2.0,
        **kwargs,
    ):
        ego_obs_dim = 6
        ref_obs_dim = 4
        obstacle_obs_dim = 4
        self.obs_dim = ego_obs_dim + ref_obs_dim * pre_horizon + obstacle_obs_dim
        super().__init__(
            obs_lower_bound=None,
            obs_upper_bound=None,
            action_lower_bound=[-max_steer, -3],
            action_upper_bound=[max_steer, 3],
            device=device,
        )
        self.robot_model = Veh3DoFModel()
        self.pre_horizon = pre_horizon
        self.veh_length = veh_length
        self.veh_width = veh_width

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
        
        current_constraint = state.context_state.constraint[:, t]
        surr_x_tf, surr_y_tf, surr_phi_tf = ego_vehicle_coordinate_transform(
            state.robot_state[:, 0],
            state.robot_state[:, 1],
            state.robot_state[:, 2],
            current_constraint[..., 0],
            current_constraint[..., 1],
            current_constraint[..., 2],
        )
        surr_u_tf = current_constraint[..., 3]
        # print("=======model========")
        # print("surr_x_tf: ", surr_x_tf)
        # print("surr_y_tf: ", surr_y_tf)
        # print("surr_phi_tf: ", surr_phi_tf)
        # print("surr_u_tf: ", surr_u_tf)
        surr_obs = torch.stack((surr_x_tf, surr_y_tf, surr_phi_tf, surr_u_tf), 1) \
            .reshape(ego_obs.shape[0], -1)
        return torch.concat((ego_obs, ref_obs, surr_obs), 1)

    def get_constraint(self, state: State) -> torch.Tensor:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.veh_width
        ego_obs = state.robot_state
        x, y, phi = ego_obs[:, :3].split(1, dim=1)
        ego_center = torch.stack(
            (
                torch.cat((x + d * torch.cos(phi), y + d * torch.sin(phi)), dim=1),
                torch.cat((x - d * torch.cos(phi), y - d * torch.sin(phi)), dim=1),
            ),
            dim=1,
        )

        surr_state = state.context_state.index_by_t().constraint
        surr_x, surr_y, surr_phi = surr_state[..., :3].split(1, dim=2)
        surr_center = torch.stack(
            (
                torch.cat(
                    (
                        surr_x + d * torch.cos(surr_phi),
                        surr_y + d * torch.sin(surr_phi),
                    ),
                    dim=2,
                ),
                torch.cat(
                    (
                        surr_x - d * torch.cos(surr_phi),
                        surr_y - d * torch.sin(surr_phi),
                    ),
                    dim=2,
                ),
            ),
            dim=2,
        )

        min_dist = np.finfo(np.float32).max * torch.ones_like(x)
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of surrounding vehicles
                dist = torch.linalg.norm(
                    ego_center[:, i].unsqueeze(1) - surr_center[..., j, :], dim=2
                )
                min_dist = torch.minimum(
                    min_dist, torch.min(dist, dim=1, keepdim=True).values
                )

        ego_to_veh_violation = 2 * r - min_dist
        # # road boundary violation
        # ego_upper_y = ego_center[:, :, 1] + r
        # ego_lower_y = ego_center[:, :, 1] - r
        # upper_bound_violation = torch.max(ego_upper_y - self.upper_bound, dim=1, keepdim=True).values
        # lower_bound_violation = torch.max(self.lower_bound - ego_lower_y, dim=1, keepdim=True).values
        # constraint = torch.cat(
        #     (ego_to_veh_violation, upper_bound_violation, lower_bound_violation), dim=1
        # )
        return ego_to_veh_violation
    
    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        ego_obs = state.robot_state
        x, y, phi, u, w = ego_obs[:, 0], ego_obs[:, 1], ego_obs[:, 2], ego_obs[:, 3], ego_obs[:, 5]
        ref_obs = state.context_state.index_by_t().reference
        ref_x, ref_y, ref_phi, ref_u = ref_obs[:, 0], ref_obs[:, 1], ref_obs[:, 2], ref_obs[:, 3]
        steer, a_x = action[:, 0], action[:, 1]
        reward = -(
            0.04 * (x - ref_x) ** 2
            + 0.04 * (y - ref_y) ** 2
            + 0.02 * angle_normalize(phi - ref_phi) ** 2
            + 0.02 * (u - ref_u) ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
            + 0.01 * a_x ** 2
        )
        return reward

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


def env_model_creator(**kwargs) -> Veh3DoFTrackingSurrCstrModel:
    return Veh3DoFTrackingSurrCstrModel(**kwargs)

