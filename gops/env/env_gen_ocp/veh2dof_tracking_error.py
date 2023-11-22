from typing import Dict, Optional

import numpy as np
from gops.env.env_gen_ocp.context.ref_traj_err import RefTrajErrContext
from gops.env.env_gen_ocp.veh2dof_tracking import Veh2DoFTracking


class Veh2DoFTrackingError(Veh2DoFTracking):
    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        dt: float = 0.1,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_steer: float = np.pi / 6,
        y_error_tol: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            dt=dt,
            path_para=path_para,
            u_para=u_para,
            max_steer=max_steer,
        )
        self.context: RefTrajErrContext = RefTrajErrContext(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_para,
            speed_param=u_para,
            y_error_tol=y_error_tol,
        )

    def _get_constraint(self) -> np.ndarray:
        y = self.robot.state[0]
        y_ref = self.context.state.reference[0, 1]
        y_error_tol = self.context.state.constraint[0]
        constraint = np.array([abs(y - y_ref) - y_error_tol], dtype=np.float32)
        return constraint


def env_creator(**kwargs):
    return Veh2DoFTrackingError(**kwargs)
