#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF data environment with surrounding vehicles constraint

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import gym

from gops.env.env_ocp.pyth_veh3dofconti_data import SimuVeh3dofconti, angle_normalize


@dataclass
class SurrVehicleData:
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0
    u: float = 0.0
    delta: float = 0.0  # front wheel angle
    l: float = 3.0  # distance from front axle to rear axle
    dt: float = 0.1

    def step(self):
        self.x = self.x + self.u * np.cos(self.phi) * self.dt
        self.y = self.y + self.u * np.sin(self.phi) * self.dt
        self.phi = self.phi + self.u * np.tan(self.delta) / self.l * self.dt
        self.phi = angle_normalize(self.phi)


class SimuVeh3dofcontiSurrCstr(SimuVeh3dofconti):
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        surr_veh_num: int = 4,
        veh_length: float = 4.8,
        veh_width: float = 2.0,
        **kwargs,
    ):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim + self.pre_horizon * 2 + surr_veh_num * 4,), 
            dtype=np.float32)
        self.surr_veh_num = surr_veh_num
        self.surr_vehs: List[SurrVehicleData] = None
        self.surr_state = np.zeros((surr_veh_num, 5), dtype=np.float32)
        self.veh_length = veh_length
        self.veh_width = veh_width
        self.info_dict.update({
            "surr_state": {"shape": (surr_veh_num, 5), "dtype": np.float32},
            "constraint": {"shape": (1,), "dtype": np.float32},
        })

    def reset(
        self,
        init_state: Optional[Sequence] = None,
        ref_time: Optional[float] = None,
        path_num: Optional[int] = None,
        u_num: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, dict]:
        super().reset(init_state, ref_time, path_num, u_num, **kwargs)

        surr_x0, surr_y0 = self.ref_points[0, :2]
        if self.path_num == 3:  # circle path
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
                delta_lon = 10 * self.np_random.uniform(-1, 1)
                delta_lat = 5 * self.np_random.uniform(-1, 1)
                if abs(delta_lon) > 0 or abs(delta_lat) > 0:
                    break
            surr_x = surr_x0 + delta_lon * np.cos(surr_phi) - delta_lat * np.sin(surr_phi)
            surr_y = surr_y0 + delta_lon * np.sin(surr_phi) + delta_lat * np.cos(surr_phi)
            surr_u = 5 + self.np_random.uniform(-1, 1)
            self.surr_vehs.append(SurrVehicleData(
                x=surr_x, y=surr_y, phi=surr_phi, u=surr_u, delta=surr_delta, dt=self.dt))
        self.update_surr_state()

        return self.get_obs(), self.info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, _ = super().step(action)

        for surr_veh in self.surr_vehs:
            surr_veh.step()
        self.update_surr_state()

        return self.get_obs(), reward, done, self.info

    def update_surr_state(self):
        for i, surr_veh in enumerate(self.surr_vehs):
            self.surr_state[i] = np.array([
                surr_veh.x, surr_veh.y, surr_veh.phi, surr_veh.u, surr_veh.delta
            ], dtype=np.float32)

    def get_obs(self) -> np.ndarray:
        obs = super().get_obs()
        surr_obs = self.surr_state[:, :4] - self.state[np.newaxis, :4]
        return np.concatenate((obs, surr_obs.flatten()))

    def get_constraint(self) -> np.ndarray:
        # collision detection using bicircle model
        d = (self.veh_length - self.veh_width) / 2  # distance from vehicle center to front/rear circle center
        r = np.sqrt(2) / 2 * self.veh_width  # circle radius

        x, y, phi = self.state[:3]
        ego_center = np.array([
            [x + d * np.cos(phi), y + d * np.sin(phi)],
            [x - d * np.cos(phi), y - d * np.sin(phi)]
        ], dtype=np.float32)

        surr_x = self.surr_state[:, 0]
        surr_y = self.surr_state[:, 1]
        surr_phi = self.surr_state[:, 2]
        surr_center = np.stack((
            np.stack(((surr_x + d * np.cos(surr_phi)), surr_y + d * np.sin(surr_phi)), axis=1),
            np.stack(((surr_x - d * np.cos(surr_phi)), surr_y - d * np.sin(surr_phi)), axis=1)
        ), axis=1)

        min_dist = np.inf
        for i in range(2):  # front and rear circle of ego vehicle
            for j in range(2):  # front and rear circle of surrounding vehicles
                dist = np.linalg.norm(ego_center[np.newaxis, i] - surr_center[:, j], axis=1)
                min_dist = min(min_dist, np.min(dist))

        return np.array([r - min_dist], dtype=np.float32)

    @property
    def info(self):
        info = super().info
        info.update({
            "surr_state": self.surr_state.copy(),
            "constraint": self.get_constraint(),
        })
        return info


def env_creator(**kwargs):
    return SimuVeh3dofcontiSurrCstr(**kwargs)


if __name__ == '__main__':
    env = SimuVeh3dofcontiSurrCstr()
    print(env.observation_space.shape)
    obs, _ = env.reset()
    print(obs.shape)
    env.step([0, 0])
