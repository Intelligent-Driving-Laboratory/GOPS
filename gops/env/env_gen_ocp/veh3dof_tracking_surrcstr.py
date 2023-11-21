from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces
from gops.env.env_gen_ocp.robot.veh3dof import angle_normalize
from gops.env.env_gen_ocp.context.ref_traj_surrcstr import RefTrajSurrCstrContext
from gops.env.env_gen_ocp.veh3dof_tracking import Veh3DoFTracking, ego_vehicle_coordinate_transform

class Veh3DoFTrackingSurrCstr(Veh3DoFTracking):
    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        dt: float = 0.1,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_acc: float = 3.0,
        max_steer: float = np.pi / 6,
        **kwargs,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            dt=dt,
            path_para=path_para,
            u_para=u_para,
            max_acc=max_acc,
            max_steer=max_steer,
            **kwargs,
        )
        self.init_high = np.array([2, 1, np.pi / 6, 2, 0.1, 0.1], dtype=np.float32)
        self.init_low = -self.init_high

        self.context: RefTrajSurrCstrContext = RefTrajSurrCstrContext(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_para,
            speed_param=u_para,
        )
        self.state_dim = 6
        ego_obs_dim = 6
        ref_obs_dim = 4
        veh_obs_dim = 4
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon + veh_obs_dim * self.context.surr_veh_num)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon + veh_obs_dim * self.context.surr_veh_num)),
            dtype=np.float32,
        )
        self.max_episode_steps = 100

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        return super().reset(
            seed=seed,
            options=options,
            init_state=init_state,
            ref_time=ref_time,
            ref_num=ref_num,
        )
    
    def _get_constraint(self) -> np.ndarray:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        veh_length = self.context.veh_length
        veh_width = self.context.veh_width
        d = (veh_length - veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * veh_width

        ego_x, ego_y, ego_phi = self.robot.state[:3]
        ego_center = np.array(
            [
                [ego_x + d * np.cos(ego_phi), ego_y + d * np.sin(ego_phi)],
                [ego_x - d * np.cos(ego_phi), ego_y - d * np.sin(ego_phi)],
            ],
            dtype=np.float32,
        )
        surr_x, surr_y, surr_phi = self.context.state.constraint[0, :, :3].T
        surr_center = np.stack(
            (
                np.stack(
                    ((surr_x + d * np.cos(surr_phi)), surr_y + d * np.sin(surr_phi)),
                    axis=1,
                ),
                np.stack(
                    ((surr_x - d * np.cos(surr_phi)), surr_y - d * np.sin(surr_phi)),
                    axis=1,
                ),
            ),
            axis=1,
        )
        # print("new ego_center: ", ego_center)
        # print("new surr_center: ", surr_center)

        min_dist = np.inf
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of surrounding vehicles
                dist = np.linalg.norm(
                    ego_center[np.newaxis, i] - surr_center[:, j], axis=1
                )
                min_dist = min(min_dist, np.min(dist))
        ego_to_veh_violation = 2 * r - min_dist
        # print("new ego_to_veh_violation: ", ego_to_veh_violation)
        return np.array([ego_to_veh_violation], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        # surr_obs
        surr_x_tf, surr_y_tf, surr_phi_tf = ego_vehicle_coordinate_transform(
            self.robot.state[0],
            self.robot.state[1],
            self.robot.state[2],
            self.context.state.constraint[0, :, 0],
            self.context.state.constraint[0, :, 1],
            self.context.state.constraint[0, :, 2],
        )
        surr_obs_rel = np.concatenate((surr_x_tf, surr_y_tf, surr_phi_tf, self.context.state.constraint[0, :, 3]))
        return np.concatenate((obs, surr_obs_rel))

    def _get_reward(self, action: np.ndarray) -> float:
        x, y, phi, u, _, w = self.robot.state
        ref_x, ref_y, ref_phi, ref_u = self.context.state.reference[0]
        steer, a_x = action
        # violation = self._get_constraint()
        # threshold = -0.1
        # punish = np.maximum(violation - threshold, 0).sum()
        # if punish > 0:
        #     punish += 1.0
        return -(
            0.04 * (x - ref_x) ** 2
            + 0.04 * (y - ref_y) ** 2
            + 0.02 * angle_normalize(phi - ref_phi) ** 2
            + 0.02 * (u - ref_u) ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
            + 0.01 * a_x ** 2
        ) - 100 * self._get_terminated()

    def _get_terminated(self) -> bool:
        x, y, phi = self.robot.state[:3]
        ref_x, ref_y, ref_phi = self.context.state.reference[0, :3]
        done = (
            (np.abs(x - ref_x) > 5)
            | (np.abs(y - ref_y) > 2)
            | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
        )
        return done

    def _render(self, ax, veh_length=4.8, veh_width=2.0):
        import matplotlib.patches as pc
        super()._render(ax, veh_length, veh_width)

        # draw surrounding vehicles
        veh_length = self.context.veh_length
        veh_width = self.context.veh_width
        for i in range(self.context.surr_veh_num):
            surr_x, surr_y, surr_phi = self.context.state.constraint[0, i, :3]
            rectan_x = surr_x - veh_length / 2 * np.cos(surr_phi) + veh_width / 2 * np.sin(surr_phi)
            rectan_y = surr_y - veh_width / 2 * np.cos(surr_phi) - veh_length / 2 * np.sin(surr_phi)
            ax.add_patch(pc.Rectangle(
                (rectan_x, rectan_y), self.context.veh_length, self.context.veh_width, angle=surr_phi * 180 / np.pi,
                facecolor='w', edgecolor='k', zorder=1))

       # render self.upper_bound and self.lower_bound with solid line
        upper_x = np.linspace(-100, 200, 100)
        lower_x = upper_x
        upper_y = np.ones_like(upper_x) * self.context.upper_bound
        lower_y = np.ones_like(lower_x) * self.context.lower_bound
        ax.plot(upper_x, upper_y, "k")
        ax.plot(lower_x, lower_y, "k")


def env_creator(**kwargs):
    return Veh3DoFTrackingSurrCstr(**kwargs)
