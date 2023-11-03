from dataclasses import dataclass, fields
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.veh3dof import Veh3DoF, angle_normalize
from gops.env.env_gen_ocp.context.ref_traj_with_static_obstacle import RefTrajWithStaticObstacleContext
from gops.env.env_gen_ocp.veh3dof_tracking import Veh3DoFTracking

class Veh3DoFTrackingDetour(Veh3DoFTracking):
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
        self.init_high = np.array([1, 0.0, np.pi / 36, 2, 0.1, 0.1], dtype=np.float32)
        self.init_low = -np.array([1, 0.8, np.pi / 36, 2, 0.1, 0.1], dtype=np.float32)

        self.context: RefTrajWithStaticObstacleContext = RefTrajWithStaticObstacleContext(
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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = 9,
    ) -> Tuple[np.ndarray, dict]:
        return super().reset(
            seed=seed,
            options=options,
            init_state=init_state,
            ref_time=ref_time,
            ref_num=9,
        )
    
    def _get_constraint(self) -> np.ndarray:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        veh_length = self.context.veh_length
        veh_width = self.context.veh_width
        d = (veh_length - veh_width) / 2
        # circle radius
        r = 0.5 * veh_width

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
        violation = self._get_constraint()
        threshold = -0.1
        punish = np.maximum(violation - threshold, 0).sum()
        if (punish > 0) :
            punish += 1.0
        done = self._get_terminated()
        return - 0.01 * (
            10.0 * (x - ref_x) ** 2
            + 10.0 * (y - ref_y) ** 2
            + 500 * angle_normalize(phi - ref_phi) ** 2
            + 5.0 * (u - ref_u) ** 2
            + 1000 * w ** 2
            + 1000  * steer ** 2
            + 50  * a_x ** 2
            + 500.0 * punish
        ) + 2.0 - 100 * done

    def _get_terminated(self) -> bool:
        x, y, phi = self.robot.state[:3]
        ref_x, ref_y, ref_phi = self.context.state.reference[0, :3]
        done = (
            (np.abs(x - ref_x) > 5)
            | (np.abs(y - ref_y) > 3)
            | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
        )
        return done

    def _get_info(self) -> dict:
        return {
            **super()._get_info(),
            "constraint": self._get_constraint().copy(),
        }

    def render(self, mode="human"):
        import matplotlib.pyplot as plt

        fig = plt.figure(num=0, figsize=(6.4, 3.2))
        plt.clf()
        ego_x, ego_y = self.robot.state[:2]
        ax = plt.axes(xlim=(ego_x - 5, ego_x + 30), ylim=(ego_y - 10, ego_y + 10))
        ax.set_aspect('equal')
        
        self._render(ax)

        plt.tight_layout()

        if mode == "rgb_array":
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            plt.pause(0.01)
            return image_from_plot
        elif mode == "human":
            plt.pause(0.01)
            plt.show()

    def _render(self, ax, veh_length=4.8, veh_width=2.0):
        import matplotlib.patches as pc

        # draw ego vehicle
        ego_x, ego_y, phi = self.robot.state[:3]
        x_offset = veh_length / 2 * np.cos(phi) - veh_width / 2 * np.sin(phi)
        y_offset = veh_length / 2 * np.sin(phi) + veh_width / 2 * np.cos(phi)
        ax.add_patch(pc.Rectangle(
            (ego_x - x_offset, ego_y - y_offset), 
            veh_length, 
            veh_width, 
            angle=np.rad2deg(phi),
            facecolor='w', 
            edgecolor='r', 
            zorder=1
        ))

        # draw reference paths
        ref_x = []
        ref_y = []
        for i in np.arange(1, self.context.pre_horizon + 1):
            ref_x.append(self.context.ref_traj.compute_x(
                self.context.ref_time + i * self.context.dt, self.context.path_num, self.context.speed_num
            ))
            ref_y .append(self.context.ref_traj.compute_y(
                self.context.ref_time + i * self.context.dt, self.context.path_num, self.context.speed_num
            ))
        ax.plot(ref_x, ref_y, 'b--', lw=1, zorder=2)

        # draw texts
        left_x = ego_x - 5
        top_y = ego_y + 15
        delta_y = 2
        ego_speed = self.robot.state[3] * 3.6  # [km/h]
        ref_speed = self.context.state.reference[0, 3] * 3.6  # [km/h]
        ax.text(left_x, top_y, f'time: {self.context.ref_time:.1f}s')
        ax.text(left_x, top_y - delta_y, f'speed: {ego_speed:.1f}km/h')
        ax.text(left_x, top_y - 2 * delta_y, f'ref speed: {ref_speed:.1f}km/h')

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

def ego_vehicle_coordinate_transform(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_phi: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform absolute coordinate of ego vehicle and reference points to the ego 
    vehicle coordinate. The origin is the position of ego vehicle. The x-axis points 
    to heading angle of ego vehicle.

    Args:
        ego_x (np.ndarray): Absolution x-coordinate of ego vehicle, shape ().
        ego_y (np.ndarray): Absolution y-coordinate of ego vehicle, shape ().
        ego_phi (np.ndarray): Absolution heading angle of ego vehicle, shape ().
        ref_x (np.ndarray): Absolution x-coordinate of reference points, shape (N,).
        ref_y (np.ndarray): Absolution y-coordinate of reference points, shape (N,).
        ref_phi (np.ndarray): Absolution tangent angle of reference points, shape (N,).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Transformed x, y, phi of reference 
        points.
    """
    cos_tf = np.cos(-ego_phi)
    sin_tf = np.sin(-ego_phi)
    ref_x_tf = (ref_x - ego_x) * cos_tf - (ref_y - ego_y) * sin_tf
    ref_y_tf = (ref_x - ego_x) * sin_tf + (ref_y - ego_y) * cos_tf
    ref_phi_tf = angle_normalize(ref_phi - ego_phi)
    return ref_x_tf, ref_y_tf, ref_phi_tf


def env_creator(**kwargs):
    return Veh3DoFTrackingDetour(**kwargs)

if __name__ == "__main__":
    # test consistency with old environment

    import numpy as np
    from gops.env.env_gen_ocp.veh3dof_tracking_detour import Veh3DoFTrackingDetour
    from gops.env.env_ocp.pyth_veh3dofconti_detour import SimuVeh3dofcontiSurrCstr


    env_old = SimuVeh3dofcontiSurrCstr()
    env_new = Veh3DoFTrackingDetour()

    seed = 1
    env_old.seed(seed)
    env_new.seed(seed)
    np.random.seed(seed)

    obs_old, _ = env_old.reset()
    obs_new, _ = env_new.reset()
    print("reset obs close:", np.isclose(obs_old, obs_new).all())

    action = np.random.random(2)
    next_obs_old, reward_old, done_old, _ = env_old.step(action)
    next_obs_new, reward_new, done_new, _ = env_new.step(action)
    print("step obs close:", np.isclose(obs_old, obs_new).all())
    print("step reward close:", np.isclose(reward_old, reward_new))