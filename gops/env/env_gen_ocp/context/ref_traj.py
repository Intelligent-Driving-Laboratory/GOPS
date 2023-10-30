from typing import Dict, Optional

import numpy as np
from gops.env.env_gen_ocp.pyth_base import ContextState, Context
from gops.env.env_ocp.resources.ref_traj_data import MultiRefTrajData


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
    ) -> ContextState[np.ndarray]:
        ref_points = []
        for i in range(2 * self.pre_horizon + 1):
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

        self.state = ContextState(reference=ref_points)
        self.ref_time = ref_time
        self.path_num = path_num
        self.speed_num = speed_num
        return self.state

    def step(self) -> ContextState[np.ndarray]:
        self.ref_time = self.ref_time + self.dt

        new_ref_point = np.array([
            self.ref_traj.compute_x(
                self.ref_time + 2 * self.pre_horizon * self.dt, 
                self.path_num, self.speed_num
            ),
            self.ref_traj.compute_y(
                self.ref_time + 2 * self.pre_horizon * self.dt, 
                self.path_num, self.speed_num
            ),
            self.ref_traj.compute_phi(
                self.ref_time + 2 * self.pre_horizon * self.dt, 
                self.path_num, self.speed_num
            ),
            self.ref_traj.compute_u(
                self.ref_time + 2 * self.pre_horizon * self.dt, 
                self.path_num, self.speed_num
            ),
        ], dtype=np.float32)
        ref_points = self.state.reference.copy()
        ref_points[:-1] = ref_points[1:]
        ref_points[-1] = new_ref_point
        self.state.reference = ref_points

        return self.state

    def get_zero_state(self) -> ContextState[np.ndarray]:
        return ContextState(
            reference=np.zeros((2 * self.pre_horizon + 1, 4), dtype=np.float32),
        )
