from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, fields
from typing import Dict, Generic, Optional, Tuple, Sequence, TypeVar

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch

stateType = TypeVar('stateType', np.ndarray, torch.Tensor)

@dataclass
class ContextState(Generic[stateType]):
    reference: stateType
    constraint: stateType
    t: stateType

    @staticmethod
    def array2tensor(context_state: 'ContextState[np.ndarray]') -> 'ContextState[torch.Tensor]':
        for field in fields(context_state):
            if isinstance(getattr(context_state, field.name), np.ndarray):
                setattr(context_state, field.name, torch.tensor(getattr(context_state, field.name)))
        return context_state
    
    @staticmethod
    def tensor2array(context_state: 'ContextState[torch.Tensor]') -> 'ContextState[np.ndarray]':
        for field in fields(context_state):
            if isinstance(getattr(context_state, field.name), torch.Tensor):
                setattr(context_state, field.name, getattr(context_state, field.name).numpy())
        return context_state
    
    @classmethod
    def stack(cls, context_states: Sequence['ContextState[stateType]']) -> 'ContextState[stateType]':
        values = []
        for field in fields(context_states[0]):
            if isinstance(getattr(context_states[0], field.name), np.ndarray):
                values.append(np.stack([getattr(context_state, field.name) for context_state in context_states]))
            elif isinstance(getattr(context_states[0], field.name), torch.Tensor):
                values.append(torch.stack([getattr(context_state, field.name) for context_state in context_states]))
            # else:
            #     values.append(getattr(context_states[0], field.name))
        return cls(*values)
    
    @classmethod
    def concat(cls, context_states: Sequence['ContextState[stateType]'], dim: int = 0) -> 'ContextState[stateType]':
        values = []
        for field in fields(context_states[0]):
            if isinstance(getattr(context_states[0], field.name), np.ndarray):
                values.append(np.concatenate([getattr(context_state, field.name) for context_state in context_states], axis=dim))
            elif isinstance(getattr(context_states[0], field.name), torch.Tensor):
                values.append(torch.concat([getattr(context_state, field.name) for context_state in context_states], dim=dim))
            # else:
            #     values.append(getattr(context_states[0], field.name))
        return cls(*values)
    
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
    CONTEXT_STATE_TYPE = ContextState

    @classmethod
    def array2tensor(cls, state: 'State[np.ndarray]') -> 'State[torch.Tensor]':
        if isinstance(state.robot_state, np.ndarray):
            state.robot_state = torch.tensor(state.robot_state)
            state.context_state = cls.CONTEXT_STATE_TYPE.array2tensor(state.context_state)
        return state
    
    @classmethod
    def tensor2array(cls, state: 'State[torch.Tensor]') -> 'State[np.ndarray]':
        if isinstance(state.robot_state, torch.Tensor):
            state.robot_state = state.robot_state.numpy()
            state.context_state = cls.CONTEXT_STATE_TYPE.tensor2array(state.context_state)
        return state
    
    @classmethod
    def stack(cls, states: Sequence['State[stateType]']) -> 'State[stateType]':
        if isinstance(states[0].robot_state, np.ndarray):
            stack = np.stack
        elif isinstance(states[0].robot_state, torch.Tensor):
            stack = torch.stack
        robot_states = stack([state.robot_state for state in states])
        context_states = cls.CONTEXT_STATE_TYPE.stack([state.context_state for state in states])
        return cls(robot_states, context_states)
    
    @classmethod
    def concat(cls, states: Sequence['State[stateType]'], dim: int = 0) -> 'State[stateType]':
        if isinstance(states[0].robot_state, np.ndarray):
            robot_states = np.concatenate([state.robot_state for state in states], axis=dim)
        elif isinstance(states[0].robot_state, torch.Tensor):
            robot_states = torch.concat([state.robot_state for state in states], dim=dim)
        context_states = cls.CONTEXT_STATE_TYPE.concat([state.context_state for state in states], dim=dim)
        return cls(robot_states, context_states)

    @abstractmethod
    def get_zero_state(self, batch_size: int = 1) -> 'State[stateType]':
        ...

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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if options is None:
            options = {}

        state = options.get('state', None)
        if state is None:
            state = self._get_init_state()
        else:
            assert type(state) == State, 'Type of initial state not supported!'
        self._state = state

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self._get_reward(action)
        self._state = self._get_next_state(action)
        terminated = self._get_terminated()
        return self._get_obs(), reward, terminated, self._get_info()

    def _get_init_state(self) -> State[np.ndarray]:
        return State(
            robot_state=self.robot.reset(),
            context_state=self.context.reset()
        )

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_zero_state(self, batch_size) -> State[np.ndarray]:
        return State(
            robot_state=self.robot.get_zero_state(batch_size),
            context_state=self.context.get_zero_state(batch_size)
        )

    @property
    def additional_info(self) -> Dict[str, State[np.ndarray]]:
        return {
            "state": self._state,
        }
