from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from gops.env.env_gen_ocp.pyth_base import ContextState, Context, stateType
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData


@dataclass
class RefTrajState(ContextState[stateType]):
    path_num: stateType
    speed_num: stateType
    ref_time: stateType


class RefTrajContext(Context):
    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        dt: float = 0.1,
        path_param: Optional[Dict[str, Dict]] = None,
        speed_param: Optional[Dict[str, Dict]] = None,
    ):
        self.ref_traj = MultiRefTrajData(
            path_param=path_param,
            speed_param=speed_param,
        )
        self.pre_horizon = pre_horizon
        self.dt = dt
        self.state = None

    def reset(
        self,
        *,
        ref_time: float,
        path_num: int,
        speed_num: int,
    ) -> RefTrajState[np.ndarray]:
        ref_points = []
        for i in range(self.pre_horizon + 1):
            ref_x = self.ref_traj.compute_x(
                ref_time + i * self.dt, path_num, speed_num
            )
            ref_y = self.ref_traj.compute_y(
                ref_time + i * self.dt, path_num, speed_num
            )
            ref_phi = self.ref_traj.compute_phi(
                ref_time + i * self.dt, path_num, speed_num
            )
            ref_u = self.ref_traj.compute_u(
                ref_time + i * self.dt, path_num, speed_num
            )
            ref_points.append([ref_x, ref_y, ref_phi, ref_u])
        ref_points = np.array(ref_points, dtype=np.float32)

        self.state = RefTrajState(
            reference=ref_points,
            constraint=0.0,
            t=0,
            path_num=path_num,
            speed_num=speed_num,
            ref_time=ref_time,
        )
        return self.state

    def step(self) -> RefTrajState[np.ndarray]:
        self.state.ref_time = self.state.ref_time + self.dt

        new_ref_point = np.array([
            self.ref_traj.compute_x(
                self.state.ref_time + self.pre_horizon * self.dt, 
                self.state.path_num, self.state.speed_num
            ),
            self.ref_traj.compute_y(
                self.state.ref_time + self.pre_horizon * self.dt, 
                self.state.path_num, self.state.speed_num
            ),
            self.ref_traj.compute_phi(
                self.state.ref_time + self.pre_horizon * self.dt, 
                self.state.path_num, self.state.speed_num
            ),
            self.ref_traj.compute_u(
                self.state.ref_time + self.pre_horizon * self.dt, 
                self.state.path_num, self.state.speed_num
            ),
        ], dtype=np.float32)
        ref_points = self.state.reference.copy()
        ref_points[:-1] = ref_points[1:]
        ref_points[-1] = new_ref_point
        self.state.reference = ref_points

        return self.state

    def get_zero_state(self) -> ContextState[np.ndarray]:
        return RefTrajState(
            reference=np.zeros((self.pre_horizon + 1, 4), dtype=np.float32),
            constraint=np.array(0.0, dtype=np.float32),
            t=np.array(0, dtype=np.int8),
            path_num=np.array(0, dtype=np.int8),
            speed_num=np.array(0, dtype=np.int8),
            ref_time=np.array(0.0, dtype=np.float32),
        )
