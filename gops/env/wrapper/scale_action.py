from typing import Union

import numpy as np
import gym
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.wrapper.base import ActionWrapper, ActionModelWrapper


class ScaleActionData(ActionWrapper):
    """
    Scales action space affinely to [:attr:`min_action`, :attr:`max_action`] for data type environment.
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray, list],
        max_action: Union[float, int, np.ndarray, list],
    ):
        super().__init__(env)

        if isinstance(min_action, list):
            min_action = np.array(min_action, dtype=env.action_space.dtype)
        if isinstance(max_action, list):
            max_action = np.array(max_action, dtype=env.action_space.dtype)

        self.min_action = np.zeros_like(env.action_space.low) + min_action
        self.max_action = np.zeros_like(env.action_space.high) + max_action
        self.action_space = gym.spaces.Box(low=self.min_action, high=self.max_action)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, self.min_action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action


class ScaleActionModel(ActionModelWrapper):
    """
    Scales action space affinely to [:attr:`min_action`, :attr:`max_action`] for model type environment.
    """

    def __init__(
        self,
        model: PythBaseModel,
        min_action: Union[float, int, np.ndarray, list],
        max_action: Union[float, int, np.ndarray, list],
    ):
        super().__init__(model)

        if isinstance(min_action, np.ndarray) or isinstance(min_action, list):
            min_action = torch.as_tensor(
                min_action,
                dtype=model.action_lower_bound.dtype,
                device=model.action_lower_bound.device
            )
        if isinstance(max_action, np.ndarray) or isinstance(max_action, list):
            max_action = torch.as_tensor(
                max_action,
                dtype=model.action_lower_bound.dtype,
                device=model.action_lower_bound.device
            )

        self.min_action = torch.zeros_like(model.action_lower_bound) + min_action
        self.max_action = torch.zeros_like(model.action_upper_bound) + max_action
        self.action_lower_bound = self.min_action
        self.action_upper_bound = self.max_action

    def action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clip(action, self.min_action, self.max_action)
        low = self.model.action_lower_bound
        high = self.model.action_upper_bound
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = torch.clip(action, low, high)
        return action
