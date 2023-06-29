from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, fields
from typing import Generic, Optional, Tuple, Sequence, TypeVar


import gym
from gym.utils import seeding
import numpy as np
import torch

stateType = TypeVar('stateType', np.ndarray, torch.Tensor)

@dataclass
class ContextState(Generic[stateType]):
    reference: stateType
    constraint: stateType
    t: int

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
            else:
                values.append(getattr(context_states[0], field.name))
        return cls(*values)
    
    @classmethod
    def concat(cls, context_states: Sequence['ContextState[stateType]'], dim: int = 0) -> 'ContextState[stateType]':
        values = []
        for field in fields(context_states[0]):
            if isinstance(getattr(context_states[0], field.name), np.ndarray):
                values.append(np.concatenate([getattr(context_state, field.name) for context_state in context_states], dim=dim))
            elif isinstance(getattr(context_states[0], field.name), torch.Tensor):
                values.append(torch.concat([getattr(context_state, field.name) for context_state in context_states], dim=dim))
            else:
                values.append(getattr(context_states[0], field.name))
        return cls(*values)
    

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
            concat = np.concatenate
        elif isinstance(states[0].robot_state, torch.Tensor):
            concat = torch.concat
        robot_states = concat([state.robot_state for state in states], dim=dim)
        context_states = cls.CONTEXT_STATE_TYPE.concat([state.context_state for state in states], dim=dim)
        return cls(robot_states, context_states)


class Robot(metaclass=ABCMeta):
    def __init__(
            self, 
            robot_state_space: Sequence, 
            action_space: Sequence
    ):
        self.robot_state_space = gym.spaces.Box(low=robot_state_space[0], high=robot_state_space[1], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=action_space[0], high=action_space[1], dtype=np.float32)
        self.robot_state = None
    
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        ...
    
# TODO: Static constraint value
class Context(metaclass=ABCMeta):
    def __init__(
            self, 
            context_space: Sequence, 
            termination_penalty: float
        ):
        self.context_state_space = gym.spaces.Box(low=context_space[0], high=context_space[1], dtype=np.float32)
        self.context_state = None
    
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        ...


class Env(gym.Env, metaclass=ABCMeta):
    def __init__(
            self, 
            observation_space: Sequence):
        super(Env, self).__init__()
        
        self.robot = Robot()
        self.context = Context()
        self._state = None
        self.observation_space = gym.spaces.Box(low=observation_space[0], high=observation_space[1], dtype=np.float32)
        self.action_space = self.robot.action_space

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
            context=self.context.step(action)
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