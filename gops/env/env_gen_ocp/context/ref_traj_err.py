from typing import Dict, Optional

import numpy as np
from gops.env.env_gen_ocp.context.ref_traj import RefTrajContext
from gops.env.env_gen_ocp.pyth_base import ContextState


class RefTrajErrContext(RefTrajContext):
    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        dt: float = 0.1,
        path_param: Optional[Dict[str, Dict]] = None,
        speed_param: Optional[Dict[str, Dict]] = None,
        y_error_tol: float = 0.2,
        u_error_tol: float = 2.0,
    ):
        super().__init__(
            pre_horizon=pre_horizon,
            dt=dt,
            path_param=path_param,
            speed_param=speed_param,
        )
        self.y_error_tol = y_error_tol
        self.u_error_tol = u_error_tol

    def reset(
        self,
        *,
        ref_time: float,
        path_num: int,
        speed_num: int,
    ) -> ContextState[np.ndarray]:
        super().reset(
            ref_time=ref_time,
            path_num=path_num,
            speed_num=speed_num,
        )
        self.state.constraint = np.array([self.y_error_tol, self.u_error_tol], dtype=np.float32)
        return self.state

    def get_zero_state(self) -> ContextState[np.ndarray]:
        state = super().get_zero_state()
        state.constraint = np.zeros((2,), dtype=np.float32)
        return state
