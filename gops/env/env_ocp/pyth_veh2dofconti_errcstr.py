#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: vehicle 2DOF data environment with tracking error constraint
#  Update: 2022-11-14, Yujie Yang: create environment

from typing import Any, Dict, Optional

import numpy as np

from gops.env.env_ocp.pyth_veh2dofconti import SimuVeh2dofconti


class SimuVeh2dofcontiErrCstr(SimuVeh2dofconti):
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        y_error_tol: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        self.y_error_tol = y_error_tol
        self.info_dict.update(
            {"constraint": {"shape": (1,), "dtype": np.float32},}
        )

    def get_constraint(self) -> np.ndarray:
        y = self.state[0]
        y_ref = self.ref_points[0, 0]
        constraint = np.array([abs(y - y_ref) - self.y_error_tol], dtype=np.float32)
        return constraint

    @property
    def info(self):
        info = super().info
        info.update(
            {"constraint": self.get_constraint(),}
        )
        return info


def env_creator(**kwargs):
    return SimuVeh2dofcontiErrCstr(**kwargs)
