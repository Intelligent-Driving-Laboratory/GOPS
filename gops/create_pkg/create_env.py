#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import gym
import numpy as np
from gym.wrappers.time_limit import TimeLimit
from gops.create_pkg.create_env_model import register as register_env_model
from gops.env.vector.sync_vector_env import SyncVectorEnv
from gops.env.vector.async_vector_env import AsyncVectorEnv
from gops.env.wrapper.action_repeat import ActionRepeatData
from gops.env.wrapper.convert_type import ConvertType
from gops.env.wrapper.gym2gymnasium import Gym2Gymnasium
from gops.env.wrapper.noise_observation import NoiseData
from gops.env.wrapper.reset_info import ResetInfoData
from gops.env.wrapper.scale_action import ScaleActionData
from gops.env.wrapper.scale_observation import ScaleObservationData
from gops.env.wrapper.shaping_reward import ShapingRewardData
from gops.env.wrapper.unify_state import StateData
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

    # print(registry.keys())
    # if new_spec.env_id in registry:
    #     print(f"Overriding environment {new_spec.env_id} already in registry.")
    
    registry[new_spec.env_id] = new_spec


# regist env and env model
env_dir_list = [e for e in os.listdir(env_path) if e.startswith("env_")]

for env_dir_name in env_dir_list:
    env_dir_abs_path = os.path.join(env_path, env_dir_name)
    file_list = os.listdir(env_dir_abs_path)
    for file in file_list:
        if file.endswith(".py") and file[0] != "_":
            try:
                env_id = file[:-3]
                mdl = importlib.import_module(f"gops.env.{env_dir_name}.{env_id}")
                env_id_camel = underline2camel(env_id)
            
                if hasattr(mdl, "env_creator"):
                    register(env_id=env_id, entry_point=getattr(mdl, "env_creator"))
                elif hasattr(mdl, env_id_camel):
                    register(env_id=env_id, entry_point=getattr(mdl, env_id_camel))
                else:
                    print(f"env {env_id} has no env_creator or {env_id_camel} in {env_dir_name}")
            except:
                RuntimeError(f"Register env {env_id} failed")

    env_model_path = os.path.join(env_path, env_dir_name, "env_model")
    if not os.path.exists(env_model_path):
        continue
    file_list = os.listdir(env_model_path)
    for file in file_list:
        if file.endswith(".py") and file[0] != "_":
            env_id = file[:-3]
            mdl = importlib.import_module(f"gops.env.{env_dir_name}.env_model.{env_id}")
            env_id_camel = underline2camel(env_id)
            if hasattr(mdl, "env_model_creator"):
                register_env_model(env_id=env_id, entry_point=getattr(mdl, "env_model_creator"))
            elif hasattr(mdl, env_id_camel):
                register_env_model(env_id=env_id, entry_point=getattr(mdl, env_id_camel))
            else:
                print(f"env {env_id} has no env_model_creator or {env_id_camel} in {env_dir_name}")


def create_env(
    env_id: str,
    *,
    vector_env_num: Optional[int] = None,
    vector_env_type: Optional[str] = None,
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
    gym2gymnasium: bool = False,
    **kwargs,
) -> object:
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
    spec_ = registry.get(env_id)

    if spec_ is None:
        raise KeyError(f"No registered env with id: {env_id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        env_creator = spec_.entry_point

    else:
        raise RuntimeError(f"{spec_.env_id} registered but entry_point is not specified")

    def env_fn():
        env = env_creator(**_kwargs)

        env = ResetInfoData(env)

        _max_episode_steps = None
        if max_episode_steps is not None:
            _max_episode_steps = max_episode_steps
        elif hasattr(env, "max_episode_steps"):
            _max_episode_steps = getattr(env, "max_episode_steps")
        if _max_episode_steps is not None:
            env = TimeLimit(env, _max_episode_steps)

        if repeat_num is not None:
            env = ActionRepeatData(env, repeat_num, sum_reward)

        env = ConvertType(env)

        env = StateData(env)

        if reward_scale is not None or reward_shift is not None:
            _reward_scale = 1.0 if reward_scale is None else reward_scale
            _reward_shift = 0.0 if reward_shift is None else reward_shift
            env = ShapingRewardData(env, _reward_shift, _reward_scale)

        if obs_noise_type is not None:
            env = NoiseData(env, obs_noise_type, obs_noise_data)

        if obs_shift is not None or obs_scale is not None:
            _obs_scale = 1.0 if obs_scale is None else obs_scale
            _obs_shift = 0.0 if obs_shift is None else obs_shift
            env = ScaleObservationData(env, _obs_shift, _obs_scale)

        if action_scale and isinstance(env.action_space, gym.spaces.Box):
            env = ScaleActionData(env, min_action, max_action)
        
        if gym2gymnasium:
            env = Gym2Gymnasium(env)

        return env

    if vector_env_num is None:
        env = env_fn()
    else:
        env_fns = [env_fn] * vector_env_num
        if vector_env_type == "sync":
            env = SyncVectorEnv(env_fns)
        elif vector_env_type == "async":
            env = AsyncVectorEnv(env_fns)
        else:
            raise ValueError(f"Invalid vector_env_type {vector_env_type}!")

    print("Create environment successfully!")
    return env
