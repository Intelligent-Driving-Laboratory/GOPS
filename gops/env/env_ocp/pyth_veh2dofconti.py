#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 2DOF data environment
#  Update Date: 2022-09-21, Jiaxin Gao: create environment

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
            u=5.0,
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
        y, phi, v, w = states
        steer = actions[0]
        u = self.vehicle_params["u"]
        k_f = self.vehicle_params["k_f"]
        k_r = self.vehicle_params["k_r"]
        l_f = self.vehicle_params["l_f"]
        l_r = self.vehicle_params["l_r"]
        m = self.vehicle_params["m"]
        I_z = self.vehicle_params["I_z"]
        next_state = [
            y + delta_t * (u * phi + v),
            phi + delta_t * w,
            (
                m * v * u
                + delta_t * (l_f * k_f - l_r * k_r) * w
                - delta_t * k_f * steer * u
                - delta_t * m * u ** 2 * w
            )
            / (m * u - delta_t * (k_f + k_r)),
            (
                I_z * w * u
                + delta_t * (l_f * k_f - l_r * k_r) * v
                - delta_t * l_f * k_f * steer * u
            )
            / (I_z * u - delta_t * (l_f ** 2 * k_f + l_r ** 2 * k_r)),
        ]
        return np.array(next_state, dtype=np.float32)


class SimuVeh2dofconti(PythBaseEnv):
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
            # initial range of [delta_y, delta_phi, v, w]
            init_high = np.array([1, np.pi / 6, 0.1, 0.1], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(SimuVeh2dofconti, self).__init__(work_space=work_space, **kwargs)

        self.vehicle_dynamics = VehicleDynamicsData()
        self.ref_traj = MultiRefTrajData(path_para, u_para)

        self.state_dim = 4
        self.pre_horizon = pre_horizon
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * (self.pre_horizon + self.state_dim)),
            high=np.array([np.inf] * (self.pre_horizon + self.state_dim)),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steer]), high=np.array([max_steer]), dtype=np.float32
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
            "ref_points": {"shape": (self.pre_horizon + 1, 2), "dtype": np.float32},
            "path_num": {"shape": (), "dtype": np.uint8},
            "u_num": {"shape": (), "dtype": np.uint8},
            "ref_time": {"shape": (), "dtype": np.float32},
            "ref": {"shape": (2,), "dtype": np.float32},
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

        if path_num is not None:
            self.path_num = path_num
        else:
            self.path_num = self.np_random.choice([0, 1, 2, 3])

        if u_num is not None:
            self.u_num = u_num
        else:
            self.u_num = self.np_random.choice([1])

        ref_points = []
        for i in range(self.pre_horizon + 1):
            ref_y = self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_phi = self.ref_traj.compute_phi(
                self.t + i * self.dt, self.path_num, self.u_num
            )
            ref_points.append([ref_y, ref_phi])
        self.ref_points = np.array(ref_points, dtype=np.float32)

        if init_state is not None:
            delta_state = np.array(init_state, dtype=np.float32)
        else:
            delta_state = self.sample_initial_state()
        self.state = np.concatenate(
            (self.ref_points[0] + delta_state[:2], delta_state[2:])
        )

        return self.get_obs(), self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reward = self.compute_reward(action)

        # ground and ego vehicle coordinates change
        relative_state = self.state.copy()
        relative_state[:2] = 0
        next_relative_state = self.vehicle_dynamics.f_xu(
            relative_state, action, self.dt
        )
        y, phi = self.state[:2]
        u = self.vehicle_dynamics.vehicle_params["u"]
        next_y = y + u * np.sin(phi) * self.dt + next_relative_state[0] * np.cos(phi)
        next_phi = phi + next_relative_state[1]
        next_phi = angle_normalize(next_phi)
        self.state = np.concatenate(
            (np.array([next_y, next_phi], dtype=np.float32), next_relative_state[2:])
        )

        self.t = self.t + self.dt

        self.ref_points[:-1] = self.ref_points[1:]
        new_ref_point = np.array(
            [
                self.ref_traj.compute_y(
                    self.t + self.pre_horizon * self.dt, self.path_num, self.u_num
                ),
                self.ref_traj.compute_phi(
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
        ego_obs = np.concatenate((self.state[:2] - self.ref_points[0], self.state[2:]))
        ref_obs = (self.state[np.newaxis, :1] - self.ref_points[1:, :1]).flatten()
        return np.concatenate((ego_obs, ref_obs))

    def compute_reward(self, action: np.ndarray) -> float:
        y, phi, v, w = self.state
        ref_y, ref_phi = self.ref_points[0]
        steer = action[0]
        return -(
            0.04 * (y - ref_y) ** 2
            + 0.02 * (phi - ref_phi) ** 2
            + 0.01 * v ** 2
            + 0.01 * w ** 2
            + 0.01 * steer ** 2
        )

    def judge_done(self) -> bool:
        y, phi = self.state[:2]
        ref_y, ref_phi = self.ref_points[0]
        done = (np.abs(y - ref_y) > 2) | (np.abs(phi - ref_phi) > np.pi)
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
        ego_x = self.ref_traj.compute_x(self.t, self.path_num, self.u_num)
        ego_y = self.state[0]
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
        ego_x = self.ref_traj.compute_x(self.t, self.path_num, self.u_num)
        ego_y, phi = self.state[:2]
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
        for i in range(1, 60):
            ref_x.append(self.ref_traj.compute_x(
                self.t + i * self.dt, self.path_num, self.u_num
            ))
            ref_y .append(self.ref_traj.compute_y(
                self.t + i * self.dt, self.path_num, self.u_num
            ))
        ax.plot(ref_x, ref_y, 'b--', lw=1, zorder=2)

        # draw texts
        left_x = ego_x - 5
        top_y = ego_y + 11
        ax.text(left_x, top_y, f'time: {self.t:.1f}s')


def env_creator(**kwargs):
    """
    make env `pyth_veh2dofconti`
    """
    return SimuVeh2dofconti(**kwargs)
