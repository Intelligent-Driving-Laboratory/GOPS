from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import copy
from gops.env.env_gen_ocp.pyth_base import ContextState
from gops.env.env_gen_ocp.context.ref_traj import RefTrajContext
from gops.env.env_ocp.env_model.pyth_veh3dofconti_model import angle_normalize

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
    
class RefTrajSurrCstrContext(RefTrajContext):
    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        surr_veh_num: int = 4,
        dt: float = 0.1,
        path_param: Optional[Dict[str, Dict]] = None,
        speed_param: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_param,
            speed_param=speed_param,
        )
        self.surr_veh_num = surr_veh_num
        self.veh_length = 4.8
        self.veh_width = 2.0
        self.surr_vehs: List[SurrVehicleData] = None

        self.lane_width = 4.0
        self.upper_bound = 0.5 * self.lane_width
        self.lower_bound = -1.5 * self.lane_width

    def reset(
        self,
        *,
        ref_time: float,
        path_num: Optional[int] = None,
        speed_num: Optional[int] = None,
    ) -> ContextState[np.ndarray]:
        self.state = super().reset(
            ref_time=ref_time,
            path_num=path_num,
            speed_num=speed_num,
        )

        ref_points = self.state.reference

        surr_x0, surr_y0 = ref_points[0, :2]
        if path_num == 3:
            # circle path
            surr_phi = ref_points[0, 2]
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
                if abs(delta_lon) > 7 or abs(delta_lat) > 3:
                    break
            surr_x = (
                surr_x0 + delta_lon * np.cos(surr_phi) - delta_lat * np.sin(surr_phi)
            )
            surr_y = (
                surr_y0 + delta_lon * np.sin(surr_phi) + delta_lat * np.cos(surr_phi)
            )
            surr_u = 5 + self.np_random.uniform(-1, 1)
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
        # self.update_surr_state()

        self.state = ContextState(
            reference=ref_points,
            constraint=self.get_surr_state_pred(),
        )
        return self.state
    
    def step_surr(self, ):
        for surr_veh in self.surr_vehs:
            surr_veh.step()
    
    def get_surr_state(self):
        surr_state = np.zeros((self.surr_veh_num, 5), dtype=np.float32)
        for i, surr_veh in enumerate(self.surr_vehs):
            surr_state[i] = np.array(
                [surr_veh.x, surr_veh.y, surr_veh.phi, surr_veh.u, surr_veh.delta],
                dtype=np.float32,
            )
        return surr_state
    
    def get_surr_state_pred(self):
        surr_state_pred = np.zeros((self.pre_horizon + 1, self.surr_veh_num, 5), dtype=np.float32)
        surr_state_pred[0] = self.get_surr_state()
        surr_vehs_backup = copy.deepcopy(self.surr_vehs)
        for i in range(self.pre_horizon):
            self.step_surr()
            surr_state_pred[i + 1] = self.get_surr_state()
        self.surr_vehs = surr_vehs_backup
        return surr_state_pred

    def step(self) -> ContextState[np.ndarray]:
        super().step()
        self.step_surr()
        self.state.constraint = self.get_surr_state_pred()

        return self.state

    def get_zero_state(self) -> ContextState[np.ndarray]:
        context_state = super().get_zero_state()
        context_state.constraint = np.zeros((self.pre_horizon + 1, self.surr_veh_num, 5), dtype=np.float32)
        return context_state
