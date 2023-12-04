import math
from typing import Optional
import torch

from gops.env.env_gen_ocp.robot.lq_model import LqModel
from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.context import lq_configs
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.robot.lq_model import LqModel

MAX_BUFFER = 20100


class LqControlModel(EnvModel):
    dt: Optional[float] = 0.1
    robot_model: LqModel

    def __init__(self, config=lq_configs.config_s3a1, **kwargs):
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            init_mean = torch.as_tensor(config["init_mean"], dtype=torch.float32)
            init_std = torch.as_tensor(config["init_std"], dtype=torch.float32)
            work_space = (init_mean - 3 * init_std, init_mean + 3 * init_std)

        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)

        self.config = config
        self.max_episode_steps = config["max_step"]
        self.robot_state_dim = len(config["state_high"])
        self.robot_model = LqModel(
            config,
            [float("-inf")] * self.robot_state_dim,
            [float("inf")] * self.robot_state_dim,
        )
        self.context = lq_configs.LQContext(balanced_state=[0, 0])
        self.work_space = work_space
        self.initial_distribution = "uniform"

        self.state_low = torch.as_tensor(config["state_low"], dtype=torch.float32)
        self.state_high = torch.as_tensor(config["state_high"], dtype=torch.float32)
        self.observation_dim = len(config["state_high"])

        self.action_upper_bound = torch.as_tensor(config["action_high"], dtype=torch.float32)
        self.action_lower_bound = torch.as_tensor(config["action_low"], dtype=torch.float32)
        self.action_dim = len(config["action_high"])
        self.control_matrix = self.robot_model.K
        self.device = torch.device("cpu")

        # environment variable
        self.observation = None

        self.first_rendering = True
        self.state_buffer = torch.as_tensor(
            (MAX_BUFFER, self.observation_dim), dtype=torch.float32
        )
        self.action_buffer = torch.as_tensor(
            (MAX_BUFFER, self.action_dim), dtype=torch.float32
        )
        self.step_counter = 0
        self.num_figures = self.observation_dim + self.action_dim
        self.ncol = math.ceil(math.sqrt(self.num_figures))
        self.nrow = math.ceil(self.num_figures / self.ncol)

    def get_obs(self, state: State) -> torch.Tensor:
        return state.robot_state

    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        x_t = state.robot_state
        u_t = action
        x_t = torch.as_tensor(x_t, dtype=torch.float32, device=self.device).unsqueeze(0)
        u_t = torch.as_tensor(u_t, dtype=torch.float32, device=self.device).unsqueeze(0)

        reward_state = torch.sum(torch.pow(x_t, 2) * self.robot_model.Q, axis=-1)
        reward_action = torch.sum(torch.pow(u_t, 2) * self.robot_model.R, axis=-1)
        reward = self.robot_model.reward_scale * (
            self.robot_model.reward_shift - 1.0 * (reward_state + reward_action)
        )
        reward = reward[0]

        return reward

    def get_terminated(self, state: State) -> torch.bool:
        obs = state.robot_state
        high = self.state_high.to(self.device)
        low = self.state_low.to(self.device)
        return torch.any(obs > high) or torch.any(obs < low)


def env_creator(**kwargs):
    """
    Create an LQ environment with the given configuration.
    """
    lqc = kwargs.get("lq_config", None)
    if lqc is None:
        config = lq_configs.config_s3a1
    elif isinstance(lqc, str):
        assert hasattr(lq_configs, "config_" + lqc)
        config = getattr(lq_configs, "config_" + lqc)
    elif isinstance(lqc, dict):
        config = lqc

    else:
        raise RuntimeError("lq_config invalid")
    lq_configs.check_lq_config(config)

    return LqControlModel(config, **kwargs)

