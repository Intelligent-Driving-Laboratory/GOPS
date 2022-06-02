#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab


import logging
from gym import spaces
from gym.utils.env_checker import _check_spaces, _check_box_obs, _check_box_action, _check_returned_values  # noqa:
import numpy as np
import importlib
import gops.create_pkg.create_env as ce

logger = logging.getLogger(__name__)


def _check_all_spaces(env):
    """
    Check that the observation and action spaces adv action_space are defined
    and inherit from gym.spaces.Space.
    """
    _check_spaces(env)
    if hasattr(env, "adv_action_space"):
        assert isinstance(env.action_space, spaces.Space), "The adv action space must inherit from gym.spaces"
    else:
        pass
        # print(f"\033[0;32;40mThis env `{env}` does not specify an adv action space, please check if it is correct\033[0m")


def _check_constraint(env):
    assert isinstance(env.constraint_dim, int), "the constraint_sdim must be an int"
    assert env.constraint_dim >= 1, "the constraint_sdim must be bigger or equal to 1"
    env.reset()
    a = env.action_space.sample()
    _, _, _, info = env.step(a)
    assert "constraint" in info.keys(), "`constraint` must be a key of info"
    if isinstance(info["constraint"], (tuple, list)):
        assert len(info["constraint"]) == env.constraint_dim, "wrong constraint dimension"
    elif isinstance(info["constraint"], np.ndarray):
        assert len(info["constraint"].shape) == 1, "wrong constraint shape"
        assert info["constraint"].shape[0] == env.constraint_dim, "wrong constraint dimension"
    else:
        raise ValueError("the constrint should be a np.ndarray, list or tuple")
    pass


def check_env_file_structures(env_file_name):
    file_obj = importlib.import_module("gops.env." + env_file_name)
    env_name_camel = ce.formatter(env_file_name)
    if hasattr(file_obj, "env_creator"):
        env_class = getattr(file_obj, "env_creator")

    elif hasattr(file_obj, env_name_camel):
        env_class = getattr(file_obj, env_name_camel)
    else:
        raise RuntimeError(f"the environment `{env_file_name}` is not implemented properly")
    return env_class

def check_env0(env):
    _check_all_spaces(env)
    observation_space = env.observation_space
    action_space = env.action_space

    obs_spaces = observation_space.spaces if isinstance(observation_space, spaces.Dict) else {"": observation_space}
    for key, space in obs_spaces.items():
        if isinstance(space, spaces.Box):
            _check_box_obs(space, key)

    # Check for the action space, it may lead to hard-to-debug issues
    if isinstance(action_space, spaces.Box):
        _check_box_action(action_space)
    if isinstance(action_space, spaces.Box):
        _check_box_action(action_space)
    if hasattr(env, "adv_action_space") and isinstance(env.adv_action_space, spaces.Box):
        _check_box_action(env.adv_action_space)

    _check_returned_values(env, observation_space, action_space)

    if hasattr(env, "constraint_dim") and env.constraint_dim is not None:
        _check_constraint(env)
    else:
        pass
        # print(f"\033[0;31;40mThis env `{env}` does not specify an constraint_dim, please check if it is correct\033[0m")


def check_env(env_name):
    print(f"checking `{env_name}_data` ...")
    try:
        env_cls = check_env_file_structures(env_name + "_data")
        env = env_cls()
    except:
        print(
            f"can not create `{env_name}`, "
            f"it may because some modules are not installed, "
            f"or the environment is not implemented correctly"
        )
        return None

    check_env0(env)



if __name__ == "__main__":
    # from gops.env.pyth_carfollowing_data import env_creator
    # env = env_creator()
    # check_env(env)

    check_env("pyth_carfollowing")
