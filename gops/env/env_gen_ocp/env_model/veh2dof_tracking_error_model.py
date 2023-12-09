import torch

from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.env_model.veh2dof_tracking_model import Veh2DoFTrackingModel


class Veh2DoFTrackingErrorModel(Veh2DoFTrackingModel):
    def get_constraint(self, state: State) -> torch.Tensor:
        y = state.robot_state[:, 0]
        y_ref = state.context_state.index_by_t().reference[:, 1]
        y_error_tol = state.context_state.constraint[:, 0]
        return (torch.abs(y - y_ref) - y_error_tol).unsqueeze(1)


def env_model_creator(**kwargs) -> Veh2DoFTrackingErrorModel:
    return Veh2DoFTrackingErrorModel(**kwargs)

