#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Check data-type environment to see whether its behaviors are reasonable!
#  Update: 2022-12-05, Yuhang Zhang: create file

import logging

import gym
from gym import spaces
from gym.utils.env_checker import (
    _check_spaces,
    _check_box_obs,
    _check_box_action,
    _check_returned_values,
)  # noqa:
import numpy as np
import importlib
import gops.create_pkg.create_env as ce

logger = logging.getLogger(__name__)


def _check_all_spaces(env):
    """
    Check whether the observation and action spaces adv action_space are defined,
    from `gym.spaces.Space`
    :param env: gym.env
    :return:
    """
    _check_spaces(env)
    if hasattr(env, "adv_action_space"):
        assert isinstance(
            env.action_space, spaces.Space
        ), "The adv action space must inherit from gym.spaces"
    else:
        pass


def _check_constraint(env):
    """
    check whether constraint is defined and in right shape
    :param env:
    :return:
    """

    assert isinstance(env.constraint_dim, int), "The dim of constraint must be an int"
    assert env.constraint_dim >= 1, "The dim of constraint must be bigger or equal to 1"
    env.reset()
    a = env.action_space.sample()
    _, _, _, info = env.step(a)
    assert "constraint" in info.keys(), "`constraint` must be a key of info"
    if isinstance(info["constraint"], (tuple, list)):
        assert (
            len(info["constraint"]) == env.constraint_dim
        ), "Wrong constraint dimension"
    elif isinstance(info["constraint"], np.ndarray):
        assert len(info["constraint"].shape) == 1, "Wrong constraint shape"
        assert (
            info["constraint"].shape[0] == env.constraint_dim
        ), "Wrong constraint dimension"
    else:
        raise ValueError("The constraint should be np.ndarray, list or tuple")
    pass


def check_env_file_structures(env_file_name):
    """
    check whether the env file has all necessary elements
    :param env_file_name: env name
    :return:
    """
    try:
        for sub in ["env_archive", "env_gym", "env_matlab", "env_ocp"]:
            try:
                file_obj = importlib.import_module(
                    "gops.env." + sub + "." + env_file_name
                )
                break
            except:
                pass
    except:
        raise RuntimeError(f"Can not found env `{env_file_name}`")
    env_name_camel = ce.formatter(env_file_name)
    if hasattr(file_obj, "env_creator"):
        env_class = getattr(file_obj, "env_creator")

    elif hasattr(file_obj, env_name_camel):
        env_class = getattr(file_obj, env_name_camel)
    else:
        raise RuntimeError(
            f"The environment `{env_file_name}` is not implemented properly"
        )
    return env_class


def check_env0(env: gym.Env):
    """
    check whether env class is well defined
    :param env:  gym.Env
    :return:
    """

    _check_all_spaces(env)
    observation_space = env.observation_space
    action_space = env.action_space

    obs_spaces = (
        observation_space.spaces
        if isinstance(observation_space, spaces.Dict)
        else {"": observation_space}
    )
    for key, space in obs_spaces.items():
        if isinstance(space, spaces.Box):
            _check_box_obs(space, key)

    # Check for the action space, it may lead to hard-to-debug issues
    if isinstance(action_space, spaces.Box):
        _check_box_action(action_space)
    if isinstance(action_space, spaces.Box):
        _check_box_action(action_space)
    if hasattr(env, "adv_action_space") and isinstance(
        env.adv_action_space, spaces.Box
    ):
        _check_box_action(env.adv_action_space)

    _check_returned_values(env, observation_space, action_space)

    if hasattr(env, "constraint_dim") and env.constraint_dim is not None:
        _check_constraint(env)
    else:
        pass


def check_env(env_name: str):
    """
    check whether env is well defined
    :param env_name: env name
    :return:
    """

    print(f"checking `{env_name}_data` ...")
    try:
        env_cls = check_env_file_structures(env_name + "_data")
        env = env_cls()
    except:
        print(
            f"Can not create `{env_name}`, "
            f"It may because some modules are not installed, "
            f"or The environment is not implemented correctly! "
        )
        return None

    check_env0(env)


def simple_check_env(env_name):
    """
    check env in simple mode
    :param env_name: env name
    :return:
    """

    print(f"checking `{env_name}_data` ...")
    env_cls = check_env_file_structures(env_name + "_data")
    env = env_cls()
    env.reset()
    a = env.action_space.sample()
    env.step(a)
    print(f"Check `{env_name}_data` successfully! ")
