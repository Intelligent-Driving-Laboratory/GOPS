#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code

from dataclasses import dataclass, field
import importlib
import os
from typing import Callable, Dict, Optional, Union

import numpy as np
from gops.env.wrapper.action_repeat import ActionRepeatModel
from gops.env.wrapper.clip_action import ClipActionModel
from gops.env.wrapper.clip_observation import ClipObservationModel
from gops.env.wrapper.mask_at_done import MaskAtDoneModel
from gops.env.wrapper.scale_action import ScaleActionModel
from gops.env.wrapper.scale_observation import ScaleObservationModel
from gops.env.wrapper.shaping_reward import ShapingRewardModel
from gops.utils.gops_path import env_path, underline2camel

@dataclass
class Spec:
    env_id: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


registry: Dict[str, Spec] = {}


def register(
    env_id: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(env_id=env_id, entry_point=entry_point, kwargs=kwargs)

    # if new_spec.env_id in registry:
    #     print(f"Overriding environment {new_spec.env_id} already in registry.")
    registry[new_spec.env_id] = new_spec


def create_env_model(
    env_id: str,
    *,
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
    **kwargs,
) -> object:
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
    env_model_id = env_id + "_model"
    spec_ = registry.get(env_model_id)

    if spec_ is None:
        raise KeyError(f"No registered env with id: {env_model_id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    _kwargs["device"] = "cuda" if _kwargs.get("use_gpu", False) else "cpu"

    if callable(spec_.entry_point):
        env_model_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.env_id} registered but entry_point is not specified")

    env_model = env_model_creator(**_kwargs)

    if mask_at_done:
        env_model = MaskAtDoneModel(env_model)
    if repeat_num is not None:
        env_model = ActionRepeatModel(env_model, repeat_num, sum_reward)

    if reward_scale is not None or reward_shift is not None:
        reward_scale = 1.0 if reward_scale is None else reward_scale
        reward_shift = 0.0 if reward_shift is None else reward_shift
        env_model = ShapingRewardModel(env_model, reward_shift, reward_scale)

    if obs_shift is not None or obs_scale is not None:
        obs_scale = 1.0 if obs_scale is None else obs_scale
        obs_shift = 0.0 if obs_shift is None else obs_shift
        env_model = ScaleObservationModel(env_model, obs_shift, obs_scale)

    if clip_obs:
        env_model = ClipObservationModel(env_model)

    if clip_action:
        env_model = ClipActionModel(env_model)

    if action_scale:
        env_model = ScaleActionModel(env_model, min_action, max_action)

    return env_model

# regist env model
env_dir_list = [e for e in os.listdir(env_path) if e.startswith("env_")]

for env_dir_name in env_dir_list:
    env_model_path = os.path.join(env_path, env_dir_name, "env_model")
    if not os.path.exists(env_model_path):
        continue
    file_list = os.listdir(env_model_path)
    for file in file_list:
        if file.endswith(".py") and file[0] != "_" and "base" not in file:
            env_id = file[:-3]
            mdl = importlib.import_module(f"gops.env.{env_dir_name}.env_model.{env_id}")
            env_id_camel = underline2camel(env_id)
            if hasattr(mdl, "env_model_creator"):
                register(env_id=env_id, entry_point=getattr(mdl, "env_model_creator"))
            elif hasattr(mdl, env_id_camel):
                register(env_id=env_id, entry_point=getattr(mdl, env_id_camel))
            else:
                print(f"env {env_id} has no env_model_creator or {env_id_camel} in {env_dir_name}")