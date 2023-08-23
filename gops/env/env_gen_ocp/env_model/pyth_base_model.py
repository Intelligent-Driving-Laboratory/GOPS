from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar, Callable, Sequence, Union

import torch

from gops.env.env_gen_ocp.pyth_base import ContextState, State

S=TypeVar('S', State, ContextState, torch.Tensor)


class Model(metaclass=ABCMeta):
    @abstractmethod
    def get_next_state(self, state: S, action: torch.Tensor) -> S:
        ...


class RobotModel(Model):
    robot_state_dim: int
    robot_state_lower_bound: torch.Tensor
    robot_state_upper_bound: torch.Tensor

    @abstractmethod
    def get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ...


class ContextModel(Model):
    @abstractmethod
    def get_next_state(self, context_state: ContextState, action: torch.Tensor) -> ContextState:
        ...


class EnvModel(Model, metaclass=ABCMeta):
    dt: float
    action_dim: int
    obs_dim: int
    action_lower_bound: torch.Tensor
    action_upper_bound: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        dt: Optional[float] = None,
        obs_lower_bound: Optional[Sequence] = None,
        obs_upper_bound: Optional[Sequence] = None,
        action_lower_bound: Optional[Sequence] = None,
        action_upper_bound: Optional[Sequence] = None,
        device: Union[torch.device, str, None] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt = dt
        if obs_lower_bound is None:
            obs_lower_bound = [float("-inf")] * self.obs_dim
        if obs_upper_bound is None:
            obs_upper_bound = [float("inf")] * self.obs_dim
        if action_lower_bound is None:
            action_lower_bound = [float("-inf")] * self.action_dim
        if action_upper_bound is None:
            action_upper_bound = [float("inf")] * self.action_dim
        self.obs_lower_bound = torch.tensor(
            obs_lower_bound, dtype=torch.float32, device=device
        )
        self.obs_upper_bound = torch.tensor(
            obs_upper_bound, dtype=torch.float32, device=device
        )
        self.action_lower_bound = torch.tensor(
            action_lower_bound, dtype=torch.float32, device=device
        )
        self.action_upper_bound = torch.tensor(
            action_upper_bound, dtype=torch.float32, device=device
        )
        self.device = device

    # Define get_constraint as Callable
    # Trick for faster constraint evaluations
    # Subclass can realize it like:
    #   def get_constraint(self, obs: torch.Tensor, info: dict) -> torch.Tensor:
    #       ...
    # This function should return Tensor of shape [n] (ndim = 1),
    # each element of which will be required to be lower than or equal to 0
    get_constraint: Callable[[State], torch.Tensor] = None

    # Just like get_constraint,
    # define function returning Tensor of shape [] (ndim = 0) in subclass
    # if you need
    get_terminal_cost: Callable[[State], torch.Tensor] = None

    def get_next_state(self, state: State, action: torch.Tensor) -> State:
        return State(
            robot_state = self.robot_model.get_next_state(state.robot_state, action),
            context_state = self.context_model.get_next_state(state.context_state, action)
        )
    
    def forward(self, obs, action, done, info):
        state = info["state"]
        next_state = self.get_next_state(state, action)
        next_obs = self.get_obs(next_state)
        reward = self.get_reward(state, action)
        terminated = self.get_terminated(state)
        next_info = {}
        next_info["state"] = next_state
        return next_obs, reward, terminated, info

    @abstractmethod
    def get_obs(self, state: State) -> torch.Tensor:
        ...

    @abstractmethod
    def get_reward(state: State, action: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_terminated(state: State) -> torch.bool:
        ...

    @property
    @abstractmethod
    def StateClass(self) -> type:
        ...

    @property
    def unwrapped(self):
        return self