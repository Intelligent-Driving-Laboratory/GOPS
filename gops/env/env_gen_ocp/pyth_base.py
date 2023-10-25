from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, fields
from typing import Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union

import gym
from gym import spaces
import numpy as np
import torch

stateType = TypeVar('stateType', np.ndarray, torch.Tensor)

@dataclass
class ContextState(Generic[stateType]):
    reference: stateType
    constraint: stateType
    t: stateType

    def array2tensor(self):
        for field in fields(self):
            if isinstance(getattr(self, field.name), np.ndarray):
                setattr(self, field.name, torch.tensor(getattr(self, field.name)))
    
    def tensor2array(self):
        for field in fields(self):
            if isinstance(getattr(self, field.name), torch.Tensor):
                setattr(self, field.name, getattr(self, field.name).numpy())

    def  __getitem__(self, index):
        try:
            value = []
            for field in fields(self):
                value.append(getattr(self, field.name)[index])
            return self.__class__(*value)
        except IndexError: "ContextState cannot be indexed or index out of range!"

    def __setitem__(self, index, value):
        try:
            for field in fields(self):
                getattr(self, field.name)[index] = getattr(value, field.name)
        except IndexError: "ContextState cannot be indexed or index out of range!"
    

@dataclass
class State(Generic[stateType]):
    robot_state: stateType
    context_state: ContextState[stateType]

    @classmethod
    def array2tensor(cls, state: 'State[np.ndarray]') -> 'State[torch.Tensor]':
        if isinstance(state.robot_state, np.ndarray):
            state.robot_state = torch.tensor(state.robot_state)
            state.context_state.array2tensor()
        return state
    
    @classmethod
    def tensor2array(cls, state: 'State[torch.Tensor]') -> 'State[np.ndarray]':
        if isinstance(state.robot_state, torch.Tensor):
            state.robot_state = state.robot_state.numpy()
            state.context_state.tensor2array()
        return state
    
    @classmethod
    def stack(cls, states: Sequence['State[stateType]'], dim: int = 0) -> 'State[stateType]':
        robot_states = stack(states, "robot_state", dim)
        context_states = stack_context_state([state.context_state for state in states], dim=dim)
        return cls(robot_states, context_states)

    @classmethod
    def concat(cls, states: Sequence['State[stateType]'], dim: int = 0) -> 'State[stateType]':
        robot_states = concat(states, "robot_state", dim)
        context_states = concat_context_state([state.context_state for state in states], dim=dim)
        return cls(robot_states, context_states)

    def batch(self, batch_size: int) -> 'State[stateType]':
        robot_state = batch(self.robot_state, batch_size)
        context_state = batch_context_state(self.context_state, batch_size)
        return self.__class__(robot_state=robot_state, context_state=context_state)

    def __getitem__(self, index):
        try:
            return State(
                robot_state=self.robot_state[index],
                context_state=self.context_state[index]
            )
        except IndexError: "State cannot be indexed or index out of range!"

    def __setitem__(self, index, value):
        try:
            self.robot_state[index] = value.robot_state
            self.context_state[index] = value.context_state
        except IndexError: "State cannot be indexed or index out of range!"

    def __len__(self):
        if self.robot_state.ndim == 1:
            return 1
        else:
            return self.robot_state.shape[0]


class Robot(metaclass=ABCMeta):
    state: np.ndarray
    state_space: spaces.Box
    action_space: spaces.Box
    
    @abstractmethod
    def reset(self, state: Optional[np.ndarray]) -> np.ndarray:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> np.ndarray:
        ...

    def get_zero_state(self) -> np.ndarray:
        return np.zeros_like(self.state_space.low)


# TODO: Static constraint value
class Context(metaclass=ABCMeta):
    state: ContextState[np.ndarray]
    
    @abstractmethod
    def reset(self) -> ContextState[np.ndarray]:
        ...

    @abstractmethod
    def step(self) -> ContextState[np.ndarray]:
        ...

    @abstractmethod
    def get_zero_state(self) -> ContextState[np.ndarray]:
        ...


class Env(gym.Env, metaclass=ABCMeta):
    robot: Robot
    context: Context
    _state: State[np.ndarray]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self._get_reward(action)
        self._state = self._get_next_state(action)
        terminated = self._get_terminated()
        return self._get_obs(), reward, terminated, self._get_info()

    def _get_info(self) -> dict:
        info = {'state': self._state}
        try:
            info['cost'] = self._get_constraint()
        except NotImplementedError:
            pass
        return info

    def _get_next_state(self, action: np.ndarray) -> State[np.ndarray]:
        return State(
            robot_state=self.robot.step(action),
            context_state=self.context.step()
        )
    
    @property
    def state(self) -> State[np.ndarray]:
        return self._state
    
    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Get observation from the current state"""
        ...

    @property
    def obs(self) -> np.ndarray:
        return self._get_obs()

    @abstractmethod
    def _get_reward(self, action: np.ndarray) -> float:
        ...

    def _get_constraint(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _get_terminated(self) -> bool:
        ...

    def get_zero_state(self) -> State[np.ndarray]:
        return State(
            robot_state=self.robot.get_zero_state(),
            context_state=self.context.get_zero_state()
        )

    @property
    def additional_info(self) -> Dict[str, State[np.ndarray]]:
        return {
            "state": self.get_zero_state(),
        }


def batch(x: Union[np.ndarray, torch.Tensor], batch_size: int) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, 0).repeat(batch_size, 0)
    elif isinstance(x, torch.Tensor):
        return torch.unsqueeze(x, 0).repeat(batch_size, 0)


def stack(x: Sequence[Union[np.ndarray, torch.Tensor]], dim: int = 0) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x[0], np.ndarray):
        return np.stack(x, axis=dim)
    elif isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=dim)


def concat(x: Sequence[Union[np.ndarray, torch.Tensor]], dim: int = 0) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x[0], np.ndarray):
        return np.concatenate(x, axis=dim)
    elif isinstance(x[0], torch.Tensor):
        return torch.concat(x, dim=dim)


def batch_context_state(context_state: Sequence['ContextState[stateType]'], batch_size: int) -> 'ContextState[stateType]':
    values = []
    for field in fields(context_state):
        values.append(batch(getattr(context_state, field.name), batch_size))
    return context_state.__class__(*values)


def stack_context_state(context_states: Sequence['ContextState[stateType]'], dim: int = 0) -> 'ContextState[stateType]':
    values = []
    for field in fields(context_states[0]):
        value_seq = [getattr(e, field.name) for e in context_states]
        values.append(stack(value_seq, dim))
    return context_states[0].__class__(*values)


def concat_context_state(context_states: Sequence['ContextState[stateType]'], dim: int = 0) -> 'ContextState[stateType]':
    values = []
    for field in fields(context_states[0]):
        value_seq = [getattr(e, field.name) for e in context_states]
        values.append(concat(value_seq, dim))
    return context_states[0].__class__(*values)
