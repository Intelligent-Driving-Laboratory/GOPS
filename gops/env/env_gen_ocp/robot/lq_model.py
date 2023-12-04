from typing import Optional, Sequence
import torch
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel
from gops.env.env_gen_ocp.robot.lq import LqModel as np_LqModel
from scipy.linalg._solvers import solve_discrete_are


class LqModel(RobotModel):
    dt: Optional[float] = 0.1

    def __init__(
        self,
        config,
        robot_state_lower_bound: Optional[Sequence] = None,
        robot_state_upper_bound: Optional[Sequence] = None,
    ):
        super().__init__(
            robot_state_lower_bound=robot_state_lower_bound,
            robot_state_upper_bound=robot_state_upper_bound,
        )
        device = torch.device("cpu")
        self.time_step = config["dt"]
        self.A = torch.as_tensor(config["A"], dtype=torch.float32, device=device)
        self.B = torch.as_tensor(config["B"], dtype=torch.float32, device=device)
        self.Q = torch.as_tensor(
            config["Q"], dtype=torch.float32, device=device
        )  # diag vector
        self.R = torch.as_tensor(
            config["R"], dtype=torch.float32, device=device
        )  # diag vector
        self.time_step = config["dt"]
        self.reward_scale = config["reward_scale"]
        self.reward_shift = config["reward_shift"]
        K, P = np_LqModel(config).compute_control_matrix()
        self.K, self.P = torch.tensor(K), torch.tensor(P)

        # IA = (1 - A * dt)
        self.state_dim = self.A.shape[0]
        IA = torch.eye(self.state_dim, device=device) - self.A * self.time_step
        self.inv_IA = torch.linalg.pinv(IA)

    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        tmp = torch.mm(self.B, action.T) * self.time_step + state.T
        tmp = tmp.float()
        x_next = torch.mm(self.inv_IA, tmp).T

        return x_next
