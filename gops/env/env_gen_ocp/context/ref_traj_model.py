from typing import Optional, Dict

import torch

from gops.env.env_gen_ocp.context.ref_traj import RefTrajState
from gops.env.env_gen_ocp.env_model.pyth_base_model import ContextModel
from gops.env.env_ocp.resources.ref_traj_model import MultiRefTrajModel


class RefTrajMdl(ContextModel):
    ref_traj: MultiRefTrajModel
    ref_traj_state: RefTrajState
    pre_horizon: int
    dt: float

    def __init__(
        self,
        *,
        pre_horizon: int = 10,
        dt: float = 0.1,
        path_param: Optional[Dict[str, Dict]] = None,
        speed_param: Optional[Dict[str, Dict]] = None,
    ):
        self.ref_traj = MultiRefTrajModel(
            path_param=path_param,
            speed_param=speed_param,
        )

        self.pre_horizon = pre_horizon
        self.dt = dt

    def get_next_state(self, context_state: RefTrajState[torch.Tensor], action: torch.Tensor) -> RefTrajState:
        ref_time = context_state.ref_time + self.dt
        path_num = context_state.path_num
        speed_num = context_state.speed_num

        ref_points = context_state.reference.clone()
        ref_points[:, :-1] = context_state.reference[:, 1:]
        new_ref_points = torch.stack(
            (
                self.ref_traj.compute_x(
                    ref_time + self.pre_horizon * self.dt, path_num, speed_num
                ),
                self.ref_traj.compute_y(
                    ref_time + self.pre_horizon * self.dt, path_num, speed_num
                ),
                self.ref_traj.compute_phi(
                    ref_time + self.pre_horizon * self.dt, path_num, speed_num
                ),
                self.ref_traj.compute_u(
                    ref_time + self.pre_horizon * self.dt, path_num, speed_num
                ),
            ),
            dim=1,
        )
        ref_points[:, -1] = new_ref_points
        
        return RefTrajState(
            reference=ref_points,
            constraint=0.0,
            t=0,
            path_num=path_num,
            speed_num=speed_num,
            ref_time=ref_time,
        )
    