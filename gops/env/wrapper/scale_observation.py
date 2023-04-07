#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: data and model type environment wrappers that scale observation
#  Update: 2022-09-21, Yuhang Zhang: create scale observation wrapper
#  Update: 2022-10-27, Yujie Yang: rewrite scale observation wrapper


from __future__ import annotations

from typing import Tuple, Union

import gym
import numpy as np
import torch

from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.env.wrapper.base import ModelWrapper
from gops.utils.gops_typing import InfoDict
from gym.core import ObsType, ActType


class ScaleObservationData(gym.Wrapper):
    """Observation scale wrapper for data type environments.

    :param env: data type environment.
    :param np.ndarray|float shift: shift factor.
    :param np.ndarray|float scale: scale factor.

    obs_rescaled = (obs + shift) * scale
    info["raw_obs"] = obs.
    example: add following to example script
        parser.add_argument("--obs_scale", default=np.array([2, 2, 2, 2]))
        parser.add_argument("--obs_shift", default=np.array([0, 0, 0, 0]))
    """

    def __init__(
        self,
        env,
        shift: Union[np.ndarray, float, list] = 0.0,
        scale: Union[np.ndarray, float, list] = 1.0,
    ):
        super(ScaleObservationData, self).__init__(env)
        if isinstance(shift, list):
            shift = np.array(shift, dtype=np.float32)
        if isinstance(scale, list):
            scale = np.array(scale, dtype=np.float32)
        self.shift = shift
        self.scale = scale

    def observation(self, observation):
        return (observation + self.shift) * self.scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_scaled = self.observation(obs)
        info["raw_obs"] = obs
        return obs_scaled, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        obs_scaled = self.observation(obs)
        info["raw_obs"] = obs
        return obs_scaled, r, d, info


class ScaleObservationModel(ModelWrapper):
    """Observation scale wrapper for model type environments.

    :param PythBaseModel model: gops model type environment.
    :param torch.Tensor|float shift: shift factor.
    :param torch.Tensor|float scale: scale factor.

    obs_rescaled = (obs + shift) * scale
    info["raw_obs"] = obs.
    example: add following to example script
        parser.add_argument("--obs_scale", default=np.array([2, 2, 2, 2]))
        parser.add_argument("--obs_shift", default=np.array([0, 0, 0, 0]))
    """

    def __init__(
        self,
        model: PythBaseModel,
        shift: Union[np.ndarray, float, list] = 0.0,
        scale: Union[np.ndarray, float, list] = 1.0,
    ):
        super(ScaleObservationModel, self).__init__(model)
        if isinstance(shift, np.ndarray) or isinstance(shift, list):
            shift = torch.as_tensor(
                shift, dtype=torch.float32, device=self.model.device
            )
        if isinstance(scale, np.ndarray) or isinstance(scale, list):
            scale = torch.as_tensor(
                scale, dtype=torch.float32, device=self.model.device
            )
        self.shift = shift
        self.scale = scale

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        info: InfoDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, InfoDict]:
        unscaled_obs = obs / self.scale - self.shift
        next_obs, reward, next_done, next_info = self.model.forward(
            unscaled_obs, action, done, info
        )
        scaled_next_obs = (next_obs + self.shift) * self.scale
        return scaled_next_obs, reward, next_done, next_info
