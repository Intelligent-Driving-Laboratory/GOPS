from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.veh3dof import Veh3DoF, angle_normalize
from gops.env.env_gen_ocp.context.ref_traj import RefTrajContext


class Veh3DoFTracking(Env):
    termination_penalty = 100.0
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    
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
        self.robot: Veh3DoF = Veh3DoF(
            dt=dt,
            max_acc=max_acc,
            max_steer=max_steer,
        )
        self.context: RefTrajContext = RefTrajContext(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_para,
            speed_param=u_para,
        )

        self.state_dim = 6
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = self.robot.action_space
        self.dt = dt
        self.pre_horizon = pre_horizon

        self.max_episode_steps = 200

        self.init_high = np.array([2, 1, np.pi / 6, 2, 0.1, 0.1], dtype=np.float32)
        self.init_low = -self.init_high

        self.seed()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        if ref_time is None:
            ref_time = 20.0 * self.np_random.uniform(0.0, 1.0)

        # Calculate path num and speed num: ref_num = [0, 1, 2,..., 7]
        if ref_num is None:
            path_num = None
            speed_num = None
        else:
            path_num = int(ref_num / 2)
            speed_num = int(ref_num % 2)

        # If no ref_num, then randomly select path and speed
        if path_num is None:
            path_num = self.np_random.choice([0, 1, 2, 3])
        if speed_num is None:
            speed_num = self.np_random.choice([0, 1])

        if init_state is None:
            delta_state = self.np_random.uniform(low=self.init_low, high=self.init_high).astype(np.float32)
        else:
            delta_state = np.array(init_state, dtype=np.float32)
        context_state = self.context.reset(
            ref_time=ref_time, path_num=path_num, speed_num=speed_num)

        init_state = np.concatenate(
            (context_state.reference[0] + delta_state[:4], delta_state[4:])
        )

        self._state = State(
            robot_state=self.robot.reset(init_state),
            context_state=context_state,
        )
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.robot.state[0],
                self.robot.state[1],
                self.robot.state[2],
                self.context.state.reference[:self.pre_horizon + 1, 0],
                self.context.state.reference[:self.pre_horizon + 1, 1],
                self.context.state.reference[:self.pre_horizon + 1, 2],
            )
        ref_u_tf = self.context.state.reference[:self.pre_horizon + 1, 3] - self.robot.state[3]
        # ego_obs: [
        # delta_x, delta_y, delta_phi, delta_u, (of the first reference point)
        # v, w (of ego vehicle)
        # ]
        ego_obs = np.concatenate((
            [ref_x_tf[0], ref_y_tf[0], ref_phi_tf[0], ref_u_tf[0]],
            self.robot.state[4:],
        ))
        # ref_obs: [
        # delta_x, delta_y, delta_phi, delta_u (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def _get_reward(self, action: np.ndarray) -> float:
        x, y, phi, u, _, w = self.robot.state
        ref_x, ref_y, ref_phi, ref_u = self.context.state.reference[0]
        steer, a_x = action
        return -(
            0.04 * (x - ref_x) ** 2
            + 0.04 * (y - ref_y) ** 2
            + 0.02 * angle_normalize(phi - ref_phi) ** 2
            + 0.02 * (u - ref_u) ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
            + 0.01 * a_x ** 2
        )

    def _get_terminated(self) -> bool:
        x, y, phi = self.robot.state[:3]
        ref_x, ref_y, ref_phi = self.context.state.reference[0, :3]
        done = (
            (np.abs(x - ref_x) > 5)
            | (np.abs(y - ref_y) > 2)
            | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
        )
        return done

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
                self.context.ref_time + i * self.dt, self.context.path_num, self.context.speed_num
            ))
            ref_y .append(self.context.ref_traj.compute_y(
                self.context.ref_time + i * self.dt, self.context.path_num, self.context.speed_num
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
    return Veh3DoFTracking(**kwargs)
