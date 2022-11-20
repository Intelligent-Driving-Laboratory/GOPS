#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF data environment with tracking error constraint

from typing import Dict, Optional

import numpy as np

from gops.env.env_ocp.pyth_veh3dofconti_data import SimuVeh3dofconti


class SimuVeh3dofcontiErrCstr(SimuVeh3dofconti):
    def __init__(
        self,
        pre_horizon: int = 10,
        path_para: Optional[Dict[str, Dict]] = None,
        u_para: Optional[Dict[str, Dict]] = None,
        y_error_tol: float = 0.2,
        u_error_tol: float = 2.0,
        **kwargs,
    ):
        super().__init__(pre_horizon, path_para, u_para, **kwargs)
        self.y_error_tol = y_error_tol
        self.u_error_tol = u_error_tol
        self.info_dict.update({
            "constraint": {"shape": (2,), "dtype": np.float32},
        })

    def get_constraint(self) -> np.ndarray:
        y, u = self.state[1], self.state[3]
        y_ref = self.ref_traj.compute_y(self.t, self.path_num, self.u_num)
        u_ref = self.ref_traj.compute_u(self.t, self.path_num, self.u_num)
        constraint = np.array([
            abs(y - y_ref) - self.y_error_tol,
            abs(u - u_ref) - self.u_error_tol,
        ], dtype=np.float32)
        return constraint

    @property
    def info(self):
        info = super().info
        info.update({
            "constraint": self.get_constraint(),
        })
        return info


def env_creator(**kwargs):
    return SimuVeh3dofcontiErrCstr(**kwargs)
