from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
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
        assert isinstance(context_state.reference, np.ndarray)
        for i in range(len(context_state._fields)):
            context_state[i] = torch.tensor(context_state[i], dtype=torch.float32)
        return context_state
    
    @staticmethod
    def tensor2array(context_state: 'ContextState[torch.Tensor]') -> 'ContextState[np.ndarray]':
        assert isinstance(context_state.reference, torch.Tensor)
        for i in range(len(context_state._fields)):
            context_state[i] = context_state[i].numpy()
        return context_state
    
    @staticmethod
    def stack(context_states: Sequence['ContextState[stateType]']) -> 'ContextState[stateType]':
        values = []
        for i in range(len(context_states[0]._fields)):
            values.append(np.stack([context_state[i] for context_state in context_states]))
        return ContextState(*values)


@dataclass
class State(Generic[stateType]):
    robot_state: stateType
    context_state: ContextState[stateType]

    @staticmethod
    def array2tensor(state: 'State[np.ndarray]') -> 'State[torch.Tensor]':
        assert isinstance(state.robot_state, np.ndarray)
        state.robot_state = torch.tensor(state.robot_state, dtype=torch.float32)
        state.context_state = ContextState.array2tensor(state.context_state)
        return state
    
    @staticmethod
    def tensor2array(state: 'State[torch.Tensor]') -> 'State[np.ndarray]':
        assert isinstance(state.robot_state, torch.Tensor)
        state.robot_state = state.robot_state.numpy()
        state.context_state = ContextState.tensor2array(state.context_state)
        return state
    
    @staticmethod
    def stack(states: Sequence['State[stateType]']) -> 'State[stateType]':
        robot_states = np.stack([state.robot_state for state in states])
        context_states = ContextState.stack([state.context_state for state in states])
        return State(robot_states, context_states)



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
    
# TODO: 静态约束值
class Context(metaclass=ABCMeta):
    def __init__(
            self, 
            context_space: Sequence, 
            termination_penalty: float
        ):
        self.context_state_space = gym.spaces.Box(low=context_space[0], high=context_space[1], dtype=np.float32)
        self.termination_penalty = termination_penalty # env
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