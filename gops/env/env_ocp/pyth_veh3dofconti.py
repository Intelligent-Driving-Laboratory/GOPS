#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment
#  Update Date: 2021-05-55, Jiaxin Gao: create environment

from typing import Dict, Optional, Sequence, Tuple

import gym
import numpy as np

from gops.env.env_ocp.pyth_base_env import PythBaseEnv
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData
from gops.utils.math_utils import angle_normalize


class VehicleDynamicsData:
    def __init__(self):
        self.vehicle_params = dict(
            k_f=-128915.5,  # front wheel cornering stiffness [N/rad]
            k_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
            l_f=1.06,  # distance from CG to front axle [m]
            l_r=1.85,  # distance from CG to rear axle [m]
            m=1412.0,  # mass [kg]
            I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
            miu=1.0,  # tire-road friction coefficient
            g=9.81,  # acceleration of gravity [m/s^2]
        )
        l_f, l_r, mass, g = (
            self.vehicle_params["l_f"],
            self.vehicle_params["l_r"],
            self.vehicle_params["m"],
            self.vehicle_params["g"],
        )
        F_zf, F_zr = l_r * mass * g / (l_f + l_r), l_f * mass * g / (l_f + l_r)
        self.vehicle_params.update(dict(F_zf=F_zf, F_zr=F_zr))

    def f_xu(self, states, actions, delta_t):
        x, y, phi, u, v, w = states
        steer, a_x = actions
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
        next_state = [
            x + delta_t * (u * np.cos(phi) - v * np.sin(phi)),
            y + delta_t * (u * np.sin(phi) + v * np.cos(phi)),
            phi + delta_t * w,
            u + delta_t * a_x,
            (
                m * v * u
                + delta_t * (l_f * k_f - l_r * k_r) * w
                - delta_t * k_f * steer * u
                - delta_t * m * np.square(u) * w
            )
            / (m * u - delta_t * (k_f + k_r)),
            (
                I_z * w * u
                + delta_t * (l_f * k_f - l_r * k_r) * v
                - delta_t * l_f * k_f * steer * u
            )
            / (I_z * u - delta_t * (np.square(l_f) * k_f + np.square(l_r) * k_r)),
        ]
        next_state[2] = angle_normalize(next_state[2])
        return np.array(next_state, dtype=np.float32)


class SimuVeh3dofconti(PythBaseEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        **kwargs,
    ):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [delta_x, delta_y, delta_phi, delta_u, v, w]
            init_high = np.array([2, 1, np.pi / 6, 2, 0.1, 0.1], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(SimuVeh3dofconti, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.ref_traj = MultiRefTrajData(path_para, u_para)

        self.state_dim = 6
        self.pre_horizon = pre_horizon
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            high=np.array([np.inf] * (ego_obs_dim + ref_obs_dim * pre_horizon)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steer, -3]),
            high=np.array([max_steer, 3]),
            dtype=np.float32,
        )
        self.dt = 0.1
        self.max_episode_steps = 200

        self.state = None
        self.path_num = None
        self.u_num = None
        self.t = None
        self.ref_points = None

        self.info_dict = {
            "state": {"shape": (self.state_dim,), "dtype": np.float32},
            "ref_points": {"shape": (self.pre_horizon + 1, 4), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (4,), "dtype": np.float32},
        }

        self.seed()

    @property
    def additional_info(self) -> Dict[str, Dict]:
        return self.info_dict

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        if ref_time is not None:
            self.t = ref_time
        else:
            self.t = 20.0 * self.np_random.uniform(0.0, 1.0)

        # Calculate path num and speed num: ref_num = [0, 1, 2,..., 7]
        if ref_num is None:
            path_num = None
            u_num = None
        else:
            path_num = int(ref_num / 2)
            u_num = int(ref_num % 2)

        # If no ref_num, then randomly select path and speed
        if path_num is not None:
            self.path_num = path_num
        else:
            self.path_num = self.np_random.choice([0, 1, 2, 3])

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = self.np_random.choice([0, 1])

        ref_points = []
        for i in range(self.pre_horizon + 1):
            ref_x = self.ref_traj.compute_x(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_y = self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_phi = self.ref_traj.compute_phi(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_u = self.ref_traj.compute_u(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_points.append([ref_x, ref_y, ref_phi, ref_u])
        self.ref_points = np.array(ref_points, dtype=np.float32)

        if init_state is not None:
            delta_state = np.array(init_state, dtype=np.float32)
        else:
            delta_state = self.sample_initial_state()
        self.state = np.concatenate(
            (self.ref_points[0] + delta_state[:4], delta_state[4:])
        )

        return self.get_obs(), self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)

        self.state = self.vehicle_dynamics.f_xu(self.state, action, self.dt)

        self.t = self.t + self.dt

        self.ref_points[:-1] = self.ref_points[1:]
        new_ref_point = np.array(
            [
                self.ref_traj.compute_x(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_y(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_phi(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_u(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
            ],
            dtype=np.float32,
        )
        self.ref_points[-1] = new_ref_point

        self.done = self.judge_done()
        if self.done:
            reward = reward - 100

        return self.get_obs(), reward, self.done, self.info

    def get_obs(self) -> np.ndarray:
        ref_x_tf, ref_y_tf, ref_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.ref_points[:, 0], self.ref_points[:, 1], self.ref_points[:, 2],
            )
        ref_u_tf = self.ref_points[:, 3] - self.state[3]
        # ego_obs: [
        # delta_x, delta_y, delta_phi, delta_u, (of the first reference point)
        # v, w (of ego vehicle)
        # ]
        ego_obs = np.concatenate(
            ([ref_x_tf[0], ref_y_tf[0], ref_phi_tf[0], ref_u_tf[0]], self.state[4:]))
        # ref_obs: [
        # delta_x, delta_y, delta_phi, delta_u (of the second to last reference point)
        # ]
        ref_obs = np.stack((ref_x_tf, ref_y_tf, ref_phi_tf, ref_u_tf), 1)[1:].flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        x, y, phi, u, _, w = self.state
        ref_x, ref_y, ref_phi, ref_u = self.ref_points[0]
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

    def judge_done(self) -> bool:
        x, y, phi = self.state[:3]
        ref_x, ref_y, ref_phi = self.ref_points[0, :3]
        done = (
            (np.abs(x - ref_x) > 5)
            | (np.abs(y - ref_y) > 2)
            | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
        )
        return done

    @property
    def info(self) -> dict:
        return {
            "state": self.state.copy(),
            "ref_points": self.ref_points.copy(),
            "path_num": self.path_num,
            "u_num": self.u_num,
            "ref_time": self.t,
            "ref": self.ref_points[0].copy(),
        }

    def render(self, mode="human"):
        import matplotlib.pyplot as plt

        fig = plt.figure(num=0, figsize=(6.4, 3.2))
        plt.clf()
        ego_x, ego_y = self.state[:2]
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
        ego_x, ego_y, phi = self.state[:3]
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
        for i in np.arange(1, 60):
            ref_x.append(self.ref_traj.compute_x(
                self.t + i * self.dt, self.path_num, self.u_num
            ))
            ref_y .append(self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            ))
        ax.plot(ref_x, ref_y, 'b--', lw=1, zorder=2)

        # draw texts
        left_x = ego_x - 5
        top_y = ego_y + 15
        delta_y = 2
        ego_speed = self.state[3] * 3.6  # [km/h]
        ref_speed = self.ref_points[0, 3] * 3.6  # [km/h]
        ax.text(left_x, top_y, f'time: {self.t:.1f}s')
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
    """
    make env `pyth_veh3dofconti`
    """
    return SimuVeh3dofconti(**kwargs)
