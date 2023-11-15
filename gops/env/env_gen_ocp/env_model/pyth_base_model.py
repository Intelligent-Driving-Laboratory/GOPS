from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar, Callable, Sequence

import torch
from torch.types import Device

from gops.env.env_gen_ocp.pyth_base import ContextState, State

S=TypeVar('S', State, ContextState, torch.Tensor)


class Model(metaclass=ABCMeta):
    @abstractmethod
    def get_next_state(self, state: S, action: torch.Tensor) -> S:
        ...


class RobotModel(Model):
    dt: Optional[float] = None
    robot_state_dim: int

    def __init__(
        self,
        robot_state_lower_bound: Optional[Sequence] = None, 
        robot_state_upper_bound: Optional[Sequence] = None, 
        device: Device = None,
    ):
        if robot_state_lower_bound is None:
            robot_state_lower_bound = [float("-inf")] * self.robot_state_dim
        if robot_state_upper_bound is None:
            robot_state_upper_bound = [float("inf")] * self.robot_state_dim
        self.robot_state_lower_bound = torch.tensor(
            robot_state_lower_bound, dtype=torch.float32, device=device
        )
        self.robot_state_upper_bound = torch.tensor(
            robot_state_upper_bound, dtype=torch.float32, device=device
        )
        self.device = device

    @abstractmethod
    def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ...


class EnvModel(Model, metaclass=ABCMeta):
    dt: Optional[float] = None
    action_dim: int
    obs_dim: int
    robot_model: RobotModel

    def __init__(
        self,
        obs_lower_bound: Optional[Sequence] = None,
        obs_upper_bound: Optional[Sequence] = None,
        action_lower_bound: Optional[Sequence] = None,
        action_upper_bound: Optional[Sequence] = None,
        device: Device = None,
    ):
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
    # This function should return Tensor of shape [B, n] (ndim = 2),
    # each element of which will be required to be lower than or equal to 0
    get_constraint: Callable[[State], torch.Tensor] = None

    # Just like get_constraint,
    # define function returning Tensor of shape [B] (ndim = 1) in subclass
    # if you need
    get_terminal_cost: Callable[[State], torch.Tensor] = None

    def get_next_state(self, state: State, action: torch.Tensor) -> State:
        next_context_state = ContextState(
            reference = state.context_state.reference,
            constraint = state.context_state.constraint,
            t = state.context_state.t + 1,
        )
        return State(
            robot_state = self.robot_model.get_next_state(state.robot_state, action),
            context_state = next_context_state
        )
    
    def robot_model_get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.robot_model.get_next_state(robot_state, action)
    
    def forward(self, obs, action, done, info):
        state = info["state"]
        next_state = self.get_next_state(state, action)
        next_obs = self.get_obs(next_state)
        reward = self.get_reward(state, action)
        terminated = self.get_terminated(next_state)
        next_info = {}
        next_info["state"] = next_state
        if self.get_constraint is not None:
            next_info["constraint"] = self.get_constraint(state)
        return next_obs, reward, terminated, next_info

    @abstractmethod
    def get_obs(self, state: State) -> torch.Tensor:
        ...

    @abstractmethod
    def get_reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_terminated(self, state: State) -> torch.bool:
        ...

    @property
    def unwrapped(self):
        return self