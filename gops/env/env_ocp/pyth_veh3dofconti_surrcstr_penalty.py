#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment with surrounding vehicles constraint
#  Update: 2023-01-08, Jiaxin Gao: create environment

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np

from gops.env.env_ocp.pyth_veh3dofconti import SimuVeh3dofconti, angle_normalize, ego_vehicle_coordinate_transform


@dataclass
class SurrVehicleData:
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    u: float = 0.0
    # front wheel angle
    delta: float = 0.0
    # distance from front axle to rear axle
    l: float = 3.0
    dt: float = 0.1

    def step(self):
        self.x = self.x + self.u * np.cos(self.phi) * self.dt
        self.y = self.y + self.u * np.sin(self.phi) * self.dt
        self.phi = self.phi + self.u * np.tan(self.delta) / self.l * self.dt
        self.phi = angle_normalize(self.phi)


class SimuVeh3dofcontiSurrCstrPenalty(SimuVeh3dofconti):
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        surr_veh_num: int = 1,
        veh_length: float = 4.8,
        veh_width: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ego_obs_dim + ref_obs_dim * pre_horizon + surr_veh_num * 4,),
            dtype=np.float32,
        )
        self.surr_veh_num = surr_veh_num
        self.surr_vehs: List[SurrVehicleData] = None
        self.surr_state = np.zeros((surr_veh_num, 5), dtype=np.float32)
        self.veh_length = veh_length
        self.veh_width = veh_width
        self.info_dict.update(
            {
                "surr_state": {"shape": (surr_veh_num, 5), "dtype": np.float32},
                "constraint": {"shape": (surr_veh_num,), "dtype": np.float32},
            }
        )

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(init_state, ref_time, ref_num, **kwargs)

        if self.path_num == 3:
            # circle path
            surr_delta = -np.arctan2(SurrVehicleData.l, self.ref_traj.ref_trajs[3].r)
        else:
            surr_delta = 0.0

        self.surr_vehs = []
        for _ in range(self.surr_veh_num):
            # avoid ego vehicle

            delta_t = self.np_random.uniform(2, 10)
            surr_phi = self.ref_traj.compute_phi(self.t + delta_t, self.path_num, self.u_num)
            delta_lon = 1.0 * self.np_random.uniform(-1, 1)
            delta_lat = 1.0 * self.np_random.uniform(-1, 1)
            surr_x = self.ref_traj.compute_x(self.t + delta_t, self.path_num, self.u_num) + delta_lon
            surr_y = self.ref_traj.compute_y(self.t + delta_t, self.path_num, self.u_num) + delta_lat

            surr_u = 0
            self.surr_vehs.append(
                SurrVehicleData(
                    x=surr_x,
                    y=surr_y,
                    phi=surr_phi,
                    u=surr_u,
                    delta=surr_delta,
                    dt=self.dt,
                )
            )
        self.update_surr_state()

        return self.get_obs(), self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, _, _ = super().step(action)
        for surr_veh in self.surr_vehs:
            surr_veh.step()
        self.update_surr_state()
        done = self.judge_done()
        return self.get_obs(), reward, done, self.info

    def update_surr_state(self):
        for i, surr_veh in enumerate(self.surr_vehs):
            self.surr_state[i] = np.array(
                [surr_veh.x, surr_veh.y, surr_veh.phi, surr_veh.u, surr_veh.delta],
                dtype=np.float32,
            )

    def compute_reward(self, action: np.ndarray) -> float:
        # x, y, phi, u, _, w = self.state
        # ref_x, ref_y, ref_phi, ref_u = self.ref_points[0]
        obs = self.get_obs()
        delta_x, delta_y, delta_phi, delta_u, v, w = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
        steer, a_x = action
        # dis = circle center distance - 2 * radius
        dis = - self.get_constraint()[0]
        collision_bound = 0.5
        dis_to_tanh = np.maximum(8 - 8 * dis / collision_bound, 0)
        punish_dis = np.tanh(dis_to_tanh - 4) + 1

        return -(
                1.0 * delta_x ** 2
                + 1.0 * delta_y ** 2
                + 0.1 * delta_phi ** 2
                + 0.1 * delta_u ** 2
                + 0.5 * v ** 2
                + 0.5 * w ** 2
                + 0.5 * steer ** 2
                + 0.5 * a_x ** 2
                + 15.0 * punish_dis
        )
        # return -(
        #         0.5 * delta_x ** 2
        #         + 0.5 * delta_y ** 2
        #         + 0.1 * delta_phi ** 2
        #         + 0.1 * delta_u ** 2
        #         + 0.5 * v ** 2
        #         + 0.5 * w ** 2
        #         + 0.5 * steer ** 2
        #         + 0.5 * a_x ** 2
        #         + 15.0 * punish_dis
        # )



    def judge_done(self) -> bool:
        x, y, phi = self.state[:3]
        ref_x, ref_y, ref_phi = self.ref_points[0, :3]
        dis = - self.get_constraint()
        done = (
                # (np.abs(x - ref_x) > 10)
                (np.abs(y - ref_y) > 5)
                | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
                | (np.any(dis < 0))
        )
        # if done:
        #     print('x - ref_x = ', np.abs(x - ref_x))
        #     print('y - ref_y = ', np.abs(y - ref_y))
        #     print('np.abs(angle_normalize(phi - ref_phi) = ', np.abs(angle_normalize(phi - ref_phi)))
        #     print('dis = ', dis)
        #     print('sur_x', self.surr_state)

        return done

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        sur_x_tf, sur_y_tf, sur_phi_tf = \
            ego_vehicle_coordinate_transform(
                self.state[0], self.state[1], self.state[2],
                self.surr_state[:, 0], self.surr_state[:, 1], self.surr_state[:, 2],
            )
        sur_u_tf = self.surr_state[:, 3] - self.state[3]
        surr_obs = np.concatenate(
            (sur_x_tf, sur_y_tf, sur_phi_tf, sur_u_tf))
        # surr_obs = np.array(sur_x_tf, sur_y_tf, sur_phi_tf, sur_u_tf)
        return np.concatenate((obs, surr_obs))

    def get_constraint(self) -> np.ndarray:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = np.sqrt(2) / 2 * self.veh_width

        x, y, phi = self.state[:3]
        ego_center = np.array(
            [
                [x + d * np.cos(phi), y + d * np.sin(phi)],
                [x - d * np.cos(phi), y - d * np.sin(phi)],
            ],
            dtype=np.float32,
        )

        surr_x = self.surr_state[:, 0]
        surr_y = self.surr_state[:, 1]
        surr_phi = self.surr_state[:, 2]
        # n * 2 * 2 first 2 is front and rear circle the second 2 is x and y position
        surr_center = np.stack(
            (
                # front circle
                # n * 2 n is num of surround 2 is x and y position
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

        min_dist = np.inf * np.ones(self.surr_veh_num, dtype=np.float32)
        for i in range(2):
            # front and rear circle of ego vehicle
            for j in range(2):
                # front and rear circle of surrounding vehicles
                dist = np.linalg.norm(
                    ego_center[np.newaxis, i] - surr_center[:, j], axis=1
                )
                min_dist = np.minimum(min_dist, dist)
        # surr_veh_num dist: between ego_veh and sur_veh min dis
        return 2 * r - min_dist


    @property
    def info(self):
        info = super().info
        info.update(
            {"surr_state": self.surr_state.copy(), "constraint": self.get_constraint(),}
        )
        return info


    def _render(self, ax):
        super()._render(ax, self.veh_length, self.veh_width)
        import matplotlib.patches as pc

        # draw surrounding vehicles
        for i in range(self.surr_veh_num):
            surr_x, surr_y, surr_phi = self.surr_state[i, :3]
            ax.add_patch(pc.Rectangle(
                (surr_x - self.veh_length / 2, surr_y - self.veh_width / 2),
                self.veh_length,
                self.veh_width,
                angle=surr_phi * 180 / np.pi,
                facecolor='w',
                edgecolor='k',
                zorder=1
            ))


def env_creator(**kwargs):
    return SimuVeh3dofcontiSurrCstrPenalty(**kwargs)
