import torch

from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.env_model.veh3dof_tracking_model import Veh3DoFTrackingModel


class Veh3DoFTrackingErrModel(Veh3DoFTrackingModel):
    def get_constraint(self, state: State) -> torch.Tensor:
        y, u = state.robot_state[:, 1], state.robot_state[:, 3]
        ref = state.context_state.index_by_t().reference
        y_ref, u_ref = ref[:, 1], ref[:, 3]
        y_error_tol = state.context_state.constraint[:, 0]
        u_error_tol = state.context_state.constraint[:, 1]
        constraint = torch.stack((
            torch.abs(y - y_ref) - y_error_tol,
            torch.abs(u - u_ref) - u_error_tol,
        ), dim=1)
        return constraint


def env_model_creator(**kwargs) -> Veh3DoFTrackingErrModel:
    return Veh3DoFTrackingErrModel(**kwargs)


if __name__ == "__main__":
    from gops.env.env_gen_ocp.veh3dof_tracking_err import Veh3DoFTrackingErr
    from gops.env.inspector.consistency_checker import check_env_model_consistency

    env = Veh3DoFTrackingErr()
    model = Veh3DoFTrackingErrModel()
    check_env_model_consistency(env, model)
