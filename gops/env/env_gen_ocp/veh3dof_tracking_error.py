from typing import Dict, Optional

import numpy as np
from gops.env.env_gen_ocp.context.ref_traj_err import RefTrajErrContext
from gops.env.env_gen_ocp.veh3dof_tracking import Veh3DoFTracking


class Veh3DoFTrackingError(Veh3DoFTracking):
    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        dt: float = 0.1,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        max_acc: float = 3.0,
        max_steer: float = np.pi / 6,
        y_error_tol: float = 0.2,
        u_error_tol: float = 2.0,
        **kwargs,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            dt=dt,
            path_para=path_para,
            u_para=u_para,
            max_acc=max_acc,
            max_steer=max_steer,
        )
        self.context: RefTrajErrContext = RefTrajErrContext(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_para,
            speed_param=u_para,
            y_error_tol=y_error_tol,
            u_error_tol=u_error_tol,
        )

    def _get_constraint(self) -> np.ndarray:
        y, u = self.robot.state[1], self.robot.state[3]
        y_ref, u_ref = self.context.state.reference[0, 1], self.context.state.reference[0, 3]
        y_error_tol, u_error_tol = self.context.state.constraint
        constraint = np.array([
            abs(y - y_ref) - y_error_tol,
            abs(u - u_ref) - u_error_tol,
        ], dtype=np.float32)
        return constraint


def env_creator(**kwargs):
    return Veh3DoFTrackingError(**kwargs)
