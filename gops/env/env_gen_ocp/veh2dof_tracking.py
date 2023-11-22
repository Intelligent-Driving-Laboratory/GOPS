from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gym import spaces
from gops.env.env_gen_ocp.pyth_base import Env, State
from gops.env.env_gen_ocp.robot.veh2dof import Veh2DoF, angle_normalize
from gops.env.env_gen_ocp.context.ref_traj import RefTrajContext


class Veh2DoFTracking(Env):
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
        max_steer: float = np.pi / 6,
        **kwargs,
    ):
        self.robot: Veh2DoF = Veh2DoF(
            dt=dt,
            max_steer=max_steer,
        )
        self.context: RefTrajContext = RefTrajContext(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_para,
            speed_param=u_para,
        )

        self.state_dim = 4
        ego_obs_dim = 4
        ref_obs_dim = 1
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = self.robot.action_space
        self.dt = dt
        self.pre_horizon = pre_horizon

        self.max_episode_steps = 200

        self.init_high = np.array([1.0, np.pi / 6, 0.1, 0.1], dtype=np.float32)
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
            speed_num = self.np_random.choice([1])

        context_state = self.context.reset(
            ref_time=ref_time, path_num=path_num, speed_num=speed_num)

        if init_state is None:
            delta_state = self.np_random.uniform(low=self.init_low, high=self.init_high).astype(np.float32)
        else:
            delta_state = np.array(init_state, dtype=np.float32)
        init_state = np.concatenate(
            (context_state.reference[0, 1:3] + delta_state[:2], delta_state[2:])
        )

        self._state = State(
            robot_state=self.robot.reset(init_state),
            context_state=context_state,
        )
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        ego_obs = np.concatenate((self.robot.state[:2] - self.context.state.reference[0, 1:3], self.robot.state[2:]))
        ref_obs = (self.robot.state[np.newaxis, :1] - self.context.state.reference[1:self.pre_horizon + 1, 1:2]).flatten()
        return np.concatenate((ego_obs, ref_obs))

    def _get_reward(self, action: np.ndarray) -> float:
        y, phi, v, w = self.robot.state
        ref_y, ref_phi = self.context.state.reference[0, 1:3]
        steer = action[0]
        return -(
            0.04 * (y - ref_y) ** 2
            + 0.02 * angle_normalize(phi - ref_phi) ** 2
            + 0.01 * v ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
        )

    def _get_terminated(self) -> bool:
        y, phi = self.robot.state[:2]
        ref_y, ref_phi = self.context.state.reference[0, 1:3]
        return (np.abs(y - ref_y) > 2) | (np.abs(phi - ref_phi) > np.pi)

    def render(self, mode="human"):
        import matplotlib.pyplot as plt

        fig = plt.figure(num=0, figsize=(6.4, 3.2))
        plt.clf()
        ego_x = self.context.ref_traj.compute_x(
            self.context.ref_time, self.context.path_num, self.context.speed_num)
        ego_y = self.robot.state[0]
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
        ego_x = self.context.ref_traj.compute_x(
            self.context.ref_time, self.context.path_num, self.context.speed_num)
        ego_y, phi = self.robot.state[:2]
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
        top_y = ego_y + 11
        ax.text(left_x, top_y, f'time: {self.context.ref_time:.1f}s')


def env_creator(**kwargs):
    return Veh2DoFTracking(**kwargs)
