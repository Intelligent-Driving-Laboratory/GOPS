#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 3DOF data environment with surrounding vehicles constraint
#  Update: 2022-11-20, Yujie Yang: create environment

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


class SimuVeh3dofcontiDetour(SimuVeh3dofconti):
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        surr_veh_num: int = 1,
        veh_length: float = 4.8,
        veh_width: float = 2.0,
        **kwargs: Any,
    ):
        init_high = np.array([1, 0.0, np.pi / 36, 2, 0.1, 0.1], dtype=np.float32)
        init_low = -np.array([1, 0.8, np.pi / 36, 2, 0.1, 0.1], dtype=np.float32)
        work_space = np.stack((init_low, init_high))
        kwargs["work_space"] = work_space
        super().__init__(pre_horizon, path_para, u_para, max_steer, **kwargs)
        ego_obs_dim = 6
        ref_obs_dim = 4
        self.max_episode_steps = 100
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
                "constraint": {"shape": (3,), "dtype": np.float32},
            }
        )

        self.lane_width = 4.0
        self.upper_bound = 0.5 * self.lane_width
        self.lower_bound = -1.5 * self.lane_width

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        ref_num: Optional[int] = 9,
        index: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, dict]:
        # init_state = np.array([0.0, -1.0, 0.0, 0.0, 0, 0], dtype=np.float32)
        super().reset(init_state, ref_time, ref_num, **kwargs)

        surr_x0, surr_y0 = self.ref_points[0, :2]
        if self.path_num == None:
            # circle path
            surr_phi = self.ref_points[0, 2]
            surr_delta = -np.arctan2(SurrVehicleData.l, self.ref_traj.ref_trajs[3].r)
        else:
            surr_phi = 0.0
            surr_delta = 0.0

        self.surr_vehs = []
        for _ in range(self.surr_veh_num):
            # avoid ego vehicle
            while True:
                # TODO: sample position according to reference trajectory
                # delta_lon = 10 * self.np_random.uniform(1.0, 2.0)
                # delta_lat = 5 * self.np_random.uniform(-1, 1)
                delta_lon = 10 * self.np_random.uniform(1.0, 2.0)
                delta_lat = 5 * self.np_random.uniform(-1, 1)
                if abs(delta_lat) > 1.4:
                    break
            surr_x = (
                surr_x0 + 20.0
                # surr_x0 + delta_lon * np.cos(surr_phi) - delta_lat * np.sin(surr_phi)
            )
            surr_y = (
                surr_y0 + 1.0
                # surr_y0 + delta_lon * np.sin(surr_phi) + delta_lat * np.cos(surr_phi)
            )
            surr_u = 0.0
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
        done = self.judge_done()

        for surr_veh in self.surr_vehs:
            surr_veh.step()
        self.update_surr_state()

        return self.get_obs(), reward, done, self.info

    def update_surr_state(self):
        for i, surr_veh in enumerate(self.surr_vehs):
            self.surr_state[i] = np.array(
                [surr_veh.x, surr_veh.y, surr_veh.phi, surr_veh.u, surr_veh.delta],
                dtype=np.float32,
            )

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        surr_x_tf, surr_y_tf, surr_phi_tf = ego_vehicle_coordinate_transform(
            self.state[0], self.state[1], self.state[2], 
            self.surr_state[:, 0], self.surr_state[:, 1], self.surr_state[:, 2])
        surr_obs_rel = np.concatenate(
            ([surr_x_tf[0], surr_y_tf[0], surr_phi_tf[0],], self.surr_state[:, 3]))  # TODO: 多辆车 , [np.sign(surr_y_tf[0])]
        return np.concatenate((obs, surr_obs_rel))

    def get_constraint(self) -> np.ndarray:
        # collision detection using bicircle model
        # distance from vehicle center to front/rear circle center
        d = (self.veh_length - self.veh_width) / 2
        # circle radius
        r = 0.5 * self.veh_width

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

        # road boundary violation
        ego_upper_y = max(ego_center[0, 1], ego_center[1, 1]) + r
        ego_lower_y = min(ego_center[0, 1], ego_center[1, 1]) - r
        upper_bound_violation = ego_upper_y - self.upper_bound
        lower_bound_violation = self.lower_bound - ego_lower_y
        return np.array([ego_to_veh_violation], dtype=np.float32)

    def compute_reward(self, action: np.ndarray) -> float:
        x, y, phi, u, _, w = self.state
        ref_x, ref_y, ref_phi, ref_u = self.ref_points[0]
        steer, a_x = action
        violation = self.get_constraint()
        threshold = -0.1
        punish = np.maximum(violation - threshold, 0).sum()
        if (punish > 0) :
            punish += 1.0
        return - 0.01 * (
            10.0 * (x - ref_x) ** 2
            + 10.0 * (y - ref_y) ** 2
            + 500 * angle_normalize(phi - ref_phi) ** 2
            + 5.0 * (u - ref_u) ** 2
            + 1000 * w ** 2
            + 1000  * steer ** 2
            + 50  * a_x ** 2
            + 500.0 * punish
        ) + 2.0

    def judge_done(self) -> bool:
        x, y, phi = self.state[:3]
        ref_x, ref_y, ref_phi = self.ref_points[0, :3]
        done = (
            (np.abs(x - ref_x) > 10)
            | (np.abs(y - ref_y) > 10)
            | (np.abs(angle_normalize(phi - ref_phi)) > np.pi)
            # | (max(self.get_constraint()) > 1.0)
        )
        return done
    
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
        # render self.upper_bound and self.lower_bound with solid line
        upper_x = np.linspace(-100, 200, 100)
        lower_x = upper_x
        upper_y = np.ones_like(upper_x) * self.upper_bound
        lower_y = np.ones_like(lower_x) * self.lower_bound
        ax.plot(upper_x, upper_y, "k")
        ax.plot(lower_x, lower_y, "k")

        # draw surrounding vehicles
        for i in range(self.surr_veh_num):
            surr_x, surr_y, surr_phi = self.surr_state[i, :3]
            rectan_x = surr_x - self.veh_length / 2 * np.cos(surr_phi) + self.veh_width / 2 * np.sin(surr_phi)
            rectan_y = surr_y - self.veh_width / 2 * np.cos(surr_phi) - self.veh_length / 2 * np.sin(surr_phi)
            ax.add_patch(pc.Rectangle(
                (rectan_x, rectan_y), self.veh_length, self.veh_width, angle=surr_phi * 180 / np.pi,
                facecolor='w', edgecolor='k', zorder=1))
            
            # distance from vehicle center to front/rear circle center
            d = (self.veh_length - self.veh_width) / 2
            # circle radius
            r = 1.2 / 2 * self.veh_width

            x, y, phi = self.state[:3]
            surr_x = self.surr_state[:, 0]
            surr_y = self.surr_state[:, 1]
            surr_phi = self.surr_state[:, 2]
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
            ax.add_patch(pc.Circle(
                ((x + d * np.cos(phi)), y + d * np.sin(phi)), r,
                facecolor='w', edgecolor='r', zorder=1))
            ax.add_patch(pc.Circle(
                ((x - d * np.cos(phi)), y - d * np.sin(phi)), r,
                facecolor='w', edgecolor='r', zorder=1))
            ax.add_patch(pc.Circle(
                surr_center[0][0], r,
                facecolor='w', edgecolor='k', zorder=1))
            ax.add_patch(pc.Circle(
                surr_center[0][1], r,
                facecolor='w', edgecolor='k', zorder=1))

def env_creator(**kwargs):
    return SimuVeh3dofcontiDetour(**kwargs)
