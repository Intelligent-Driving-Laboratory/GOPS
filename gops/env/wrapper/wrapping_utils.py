#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: automatically wrap environment according arguments passed by script
#  Update: 2022-10-27, Yujie Yang: create wrapping utils


import numpy as np

from gym.wrappers.time_limit import TimeLimit

from gops.env.wrapper.clip_action import ClipActionModel
from gops.env.wrapper.clip_observation import ClipObservationModel
from gops.env.wrapper.convert_type import ConvertType
from gops.env.wrapper.mask_at_done import MaskAtDoneModel
from gops.env.wrapper.noise_observation import NoiseData
from gops.env.wrapper.reset_info import ResetInfoData
from gops.env.wrapper.scale_action import ScaleActionData, ScaleActionModel
from gops.env.wrapper.scale_observation import (
    ScaleObservationData,
    ScaleObservationModel,
)
from gops.env.wrapper.shaping_reward import ShapingRewardData, ShapingRewardModel
from gops.env.wrapper.unify_state import StateData
from gops.env.wrapper.action_repeat import ActionRepeatData, ActionRepeatModel

from typing import Optional, Union


def all_none(a, b):
    if (a is None) and (b is None):
        return True
    else:
        return False


def wrapping_env(
    env,
    max_episode_steps: Optional[int] = None,
    reward_shift: Optional[float] = None,
    reward_scale: Optional[float] = None,
    obs_shift: Union[np.ndarray, float, list, None] = None,
    obs_scale: Union[np.ndarray, float, list, None] = None,
    obs_noise_type: Optional[str] = None,
    obs_noise_data: Optional[list] = None,
    repeat_num: Optional[int] = None,
    sum_reward: bool = True,
    action_scale: bool = True,
    min_action: Union[float, int, np.ndarray, list] = -1.0,
    max_action: Union[float, int, np.ndarray, list] = 1.0,
):
    """Automatically wrap data type environment according to input arguments. Wrapper will not be used
        if all corresponding parameters are set to None.

    :param env: original data type environment.
    :param Optional[int] max_episode_steps: parameter for gym.wrappers.time_limit.TimeLimit wrapper.
        if it is set to None but environment has 'max_episode_steps' attribute, it will be filled in
        TimeLimit wrapper alternatively.
    :param Optional[float] reward_shift: parameter for reward shaping wrapper.
    :param Optional[float] reward_scale: parameter for reward shaping wrapper.
    :param Union[np.ndarray, float, list, None] obs_shift: parameter for observation scale wrapper.
    :param Union[np.ndarray, float, list, None] obs_scale: parameter for observation scale wrapper.
    :param Optional[str] obs_noise_type: parameter for observation noise wrapper.
    :param Optional[list] obs_noise_data: parameter for observation noise wrapper.
    :param Optional[int] repeat_num: parameter for action repeat wrapper.
    :param bool sum_reward: parameter for action repeat wrapper.
    :param bool action_scale: parameter for scale action wrapper, default to True.
    :param Union[float, int, np.ndarray, list] min_action: minimum action after scaling.
    :param Union[float, int, np.ndarray, list] max_action: maximum action after scaling.
    :return: wrapped data type environment.
    """
    env = ResetInfoData(env)
    if max_episode_steps is None and hasattr(env, "max_episode_steps"):
        max_episode_steps = getattr(env, "max_episode_steps")
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)
    if repeat_num is not None:
        env = ActionRepeatData(env, repeat_num, sum_reward)
    env = ConvertType(env)
    env = StateData(env)

    if not all_none(reward_scale, reward_shift):
        reward_scale = 1.0 if reward_scale is None else reward_scale
        reward_shift = 0.0 if reward_shift is None else reward_shift
        env = ShapingRewardData(env, reward_shift, reward_scale)

    if obs_noise_type is not None:
        env = NoiseData(env, obs_noise_type, obs_noise_data)

    if not all_none(obs_shift, obs_scale):
        obs_scale = 1.0 if obs_scale is None else obs_scale
        obs_shift = 0.0 if obs_shift is None else obs_shift
        env = ScaleObservationData(env, obs_shift, obs_scale)

    if action_scale:
        env = ScaleActionData(env, min_action, max_action)

    return env


def wrapping_model(
    model,
    reward_shift: Optional[float] = None,
    reward_scale: Optional[float] = None,
    obs_shift: Union[np.ndarray, float, list, None] = None,
    obs_scale: Union[np.ndarray, float, list, None] = None,
    clip_obs: bool = True,
    clip_action: bool = True,
    mask_at_done: bool = True,
    repeat_num: Optional[int] = None,
    sum_reward: bool = True,
    action_scale: bool = True,
    min_action: Union[float, int, np.ndarray, list] = -1.0,
    max_action: Union[float, int, np.ndarray, list] = 1.0,
):
    """Automatically wrap model type environment according to input arguments. Wrapper will not be used
        if all corresponding parameters are set to None.

    :param model: original data type environment.
    :param Optional[float] reward_shift: parameter for reward shaping wrapper.
    :param Optional[float] reward_scale: parameter for reward shaping wrapper.
    :param Union[np.ndarray, float, list, None] obs_shift: parameter for observation scale wrapper.
    :param Union[np.ndarray, float, list, None] obs_scale: parameter for observation scale wrapper.
    :param bool clip_obs: parameter for clip observation wrapper, default to True.
    :param bool clip_action: parameter for clip action wrapper, default to True.
    :param bool mask_at_done: parameter for mask at done wrapper, default to True.
    :param Optional[int] repeat_num: parameter for action repeat wrapper.
    :param bool sum_reward: parameter for action repeat wrapper.
    :param bool action_scale: parameter for scale action wrapper, default to True.
    :param Union[float, int, np.ndarray, list] min_action: minimum action after scaling.
    :param Union[float, int, np.ndarray, list] max_action: maximum action after scaling.
    :return: wrapped model type environment.
    """
    if mask_at_done:
        model = MaskAtDoneModel(model)
    if repeat_num is not None:
        model = ActionRepeatModel(model, repeat_num, sum_reward)

    if not all_none(reward_scale, reward_shift):
        reward_scale = 1.0 if reward_scale is None else reward_scale
        reward_shift = 0.0 if reward_shift is None else reward_shift
        model = ShapingRewardModel(model, reward_shift, reward_scale)

    if not all_none(obs_shift, obs_scale):
        obs_scale = 1.0 if obs_scale is None else obs_scale
        obs_shift = 0.0 if obs_shift is None else obs_shift
        model = ScaleObservationModel(model, obs_shift, obs_scale)

    if clip_obs:
        model = ClipObservationModel(model)

    if clip_action:
        model = ClipActionModel(model)

    if action_scale:
        model = ScaleActionModel(model, min_action, max_action)

    return model
