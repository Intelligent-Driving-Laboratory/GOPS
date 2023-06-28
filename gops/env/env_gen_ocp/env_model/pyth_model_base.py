from abc import ABCMeta, abstractmethod
from typing import Optional, TypeVar, Callable, Tuple

import numpy as np
import torch

from gops.env.env_gen_ocp.pyth_base import Context, State

S=TypeVar('S', State, Context)


class Model(metaclass=ABCMeta):
    @abstractmethod
    def get_next_state(self, state: S, action: torch.Tensor) -> S:
        ...


class RobotModel(Model):
    @abstractmethod
    def get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ...


class ContextModel(Model):
    @abstractmethod
    def get_next_state(self, context: Context, action: torch.Tensor) -> Context:
        ...


class EnvModel(Model, metaclass=ABCMeta):
    def __init__(
            self,
    ):
        self.robot_model = RobotModel()
        self.context_model = ContextModel()
        
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
        _state = State()
        _state.robot_state = self.robot_model.get_next_state(state.robot_state, action)
        if self.context_model is not None:
            _state.context = self.context_model.get_next_state(state.context, action)
        else:
            _state.context = state.context
        return _state

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
    def unwrapped(self):
        return self