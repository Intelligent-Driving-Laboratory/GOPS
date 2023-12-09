from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, fields
from typing import Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union
from copy import deepcopy
from gym.utils.seeding import RandomNumberGenerator

import gym
from gym import spaces
import numpy as np
import torch

stateType = TypeVar('stateType', np.ndarray, torch.Tensor)

@dataclass
class ContextState(Generic[stateType]):
    reference: stateType
    constraint: Optional[stateType] = None
    t: Union[int, stateType] = 0

    def array2tensor(self) -> 'ContextState[torch.Tensor]':
        value = []
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, np.ndarray):
                value.append(torch.from_numpy(v))
            else:
                value.append(v)
        return self.__class__(*value)

    def tensor2array(self) -> 'ContextState[np.ndarray]':
        value = []
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                value.append(v.numpy())
            else:
                value.append(v)
        return self.__class__(*value)

    def cuda(self) -> 'ContextState[torch.Tensor]':
        value = []
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                value.append(v.cuda())
            else:
                value.append(v)
        return self.__class__(*value)

    def  __getitem__(self, index):
        try:
            value = []
            for field in fields(self):
                v = getattr(self, field.name)
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    value.append(v[index])
                else:
                    value.append(v)
            return self.__class__(*value)
        except IndexError: "ContextState cannot be indexed or index out of range!"

    def __setitem__(self, index, value):
        try:
            for field in fields(self):
                v = getattr(self, field.name)
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    v[index] = getattr(value, field.name)
        except IndexError: "ContextState cannot be indexed or index out of range!"
    
    def index_by_t(self) -> 'ContextState[stateType]':
        value = []
        for field in fields(self):
            v = getattr(self, field.name)
            if field.name == "t":
                value.append(0)
            elif isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim > 2:
                value.append(v[np.arange(v.shape[0]), self.t])
            else:
                value.append(v)
        return self.__class__(*value)


@dataclass
class State(Generic[stateType]):
    robot_state: stateType
    context_state: ContextState[stateType]

    def array2tensor(self) -> 'State[torch.Tensor]':
        assert isinstance(self.robot_state, np.ndarray)
        robot_state = torch.from_numpy(self.robot_state)
        context_state = self.context_state.array2tensor()
        return self.__class__(robot_state, context_state)

    def tensor2array(self) -> 'State[np.ndarray]':
        assert isinstance(self.robot_state, torch.Tensor)
        robot_state = self.robot_state.numpy()
        context_state = self.context_state.tensor2array()
        return self.__class__(robot_state, context_state)

    def cuda(self) -> 'State[torch.Tensor]':
        assert isinstance(self.robot_state, torch.Tensor)
        robot_state = self.robot_state.cuda()
        context_state = self.context_state.cuda()
        return self.__class__(robot_state, context_state)

    @classmethod
    def stack(cls, states: Sequence['State[stateType]'], dim: int = 0) -> 'State[stateType]':
        robot_states = stack([state.robot_state for state in states], dim)
        context_states = stack_context_state([state.context_state for state in states], dim=dim)
        return cls(robot_states, context_states)

    @classmethod
    def concat(cls, states: Sequence['State[stateType]'], dim: int = 0) -> 'State[stateType]':
        robot_states = concat([state.robot_state for state in states], dim)
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
    def step(self, action: np.ndarray) -> np.ndarray:
        ...

    def reset(self, state: np.ndarray) -> np.ndarray:
        self.state = state.copy()
        return state

    def get_zero_state(self) -> np.ndarray:
        return np.zeros_like(self.state_space.low)


# TODO: Static constraint value
class Context(metaclass=ABCMeta):
    state: ContextState[np.ndarray]
    np_random: Optional[RandomNumberGenerator] = RandomNumberGenerator

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
    termination_penalty: float = 0.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self._get_reward(action)
        self._state = self._get_next_state(action)
        terminated = self._get_terminated()
        if terminated:
            reward -= self.termination_penalty
        return self._get_obs(), reward, terminated, self._get_info()

    def _get_info(self) -> dict:
        info = {'state': deepcopy(self._state)}
        try:
            info['constraint'] = self._get_constraint()
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

    def seed(self, seed=None):
        super().seed(seed)
        self.context.np_random = self.np_random

def batch(x: Union[np.ndarray, torch.Tensor], batch_size: int) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x, np.ndarray):
        if batch_size == 1:
            return np.expand_dims(x, 0)
        else:
            return np.expand_dims(x, 0).repeat(batch_size, 0)
    elif isinstance(x, torch.Tensor):
        if batch_size == 1:
            return torch.unsqueeze(x, 0)
        else:
            return torch.unsqueeze(x, 0).repeat((batch_size,) + (1,) * x.ndim)


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


def batch_context_state(context_state: 'ContextState[stateType]', batch_size: int) -> 'ContextState[stateType]':
    values = []
    for field in fields(context_state):
        v = getattr(context_state, field.name)
        if isinstance(v, (np.ndarray, torch.Tensor)):
            values.append(batch(v, batch_size))
        else:
            values.append(v)
    return context_state.__class__(*values)


def stack_context_state(context_states: Sequence['ContextState[stateType]'], dim: int = 0) -> 'ContextState[stateType]':
    values = []
    for field in fields(context_states[0]):
        v = getattr(context_states[0], field.name)
        if isinstance(v, (np.ndarray, torch.Tensor)):
            value_seq = [getattr(e, field.name) for e in context_states]
            values.append(stack(value_seq, dim))
        else:
            values.append(v)
    return context_states[0].__class__(*values)


def concat_context_state(context_states: Sequence['ContextState[stateType]'], dim: int = 0) -> 'ContextState[stateType]':
    values = []
    for field in fields(context_states[0]):
        v = getattr(context_states[0], field.name)
        if isinstance(v, (np.ndarray, torch.Tensor)):
            value_seq = [getattr(e, field.name) for e in context_states]
            values.append(concat(value_seq, dim))
        else:
            values.append(v)
    return context_states[0].__class__(*values)
