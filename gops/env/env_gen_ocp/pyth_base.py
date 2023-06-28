from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Sequence, TypeVar, Callable


import gym
from gym.utils import seeding
import numpy as np
import torch

stateType = TypeVar('stateType', np.ndarray, torch.Tensor)

@dataclass
class ContextState():
    reference: stateType
    constraint: stateType
    t: int

@dataclass
class State():
    robot_state: stateType
    context_state: ContextState


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
        self.context_space = gym.spaces.Box(low=context_space[0], high=context_space[1], dtype=np.float32)
        self.termination_penalty = termination_penalty # env
        self.context = None
    
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
        self.task = Context()
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

    def _get_init_state(self) -> State:
        return State(
            robot_state=self.robot.reset(),
            context_state=self.task.reset()
        )

    def _get_info(self) -> dict:
        if self._get_constraint == None:
            return {'state': self._state}
        else:
            return {'state': self._state, 'cost': self._get_constraint()}

    def _get_next_state(self, action: np.ndarray) -> State:
        return State(
            robot_state=self.robot.step(action),
            context=self.task.step(action)
        )
    
    @property
    def state(self) -> State:
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

    _get_constraint: Callable[[], np.ndarray] = None
    
    @abstractmethod
    def _get_terminated(self) -> bool:
        ...

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]