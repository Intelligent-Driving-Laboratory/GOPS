#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Check the correctness of model-type environment
#  Update: 2022-12-05, Yuhang Zhang: create file

import torch
import numpy as np
import importlib
import gops.create_pkg.create_env_model as ce
from gops.env.inspector.env_data_checker import check_env_file_structures


def check_env_model_file_structures(env_file_name):
    """
    check whether the env model file has all necessary elements
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
        raise RuntimeError(f"can not found env `{env_file_name}`")
    env_name_camel = ce.formatter(env_file_name)
    if hasattr(file_obj, "env_model_creator"):
        env_class = getattr(file_obj, "env_model_creator")

    elif hasattr(file_obj, env_name_camel):
        env_class = getattr(file_obj, env_name_camel)
    else:
        raise RuntimeError(
            f"The environment `{env_file_name}` is not implemented properly"
        )
    return env_class


def check_model0(env, env_model):
    """
    check whether env model class is well defined
    :param env: gym.Env
    :param env_model:
    :return:
    """

    assert hasattr(env_model, "lb_state"), "Env model must have lb_state"
    assert hasattr(env_model, "hb_state"), "Env model must have hb_state"
    assert hasattr(env_model, "lb_action"), "Env model must have lb_action"
    assert hasattr(env_model, "hb_action"), "Env model must have hb_action"

    assert isinstance(env_model.lb_action, torch.Tensor)
    assert isinstance(env_model.hb_state, torch.Tensor)
    assert isinstance(env_model.lb_action, torch.Tensor)
    assert isinstance(env_model.hb_action, torch.Tensor)

    batch_size = 10

    s = [env.reset() for _ in range(batch_size)]
    a = [env.action_space.sample() for _ in range(batch_size)]

    s = np.stack(s)
    a = np.stack(a)

    s_torch = torch.as_tensor(s, dtype=torch.float32).reshape(batch_size, -1)
    a_torch = torch.as_tensor(a, dtype=torch.float32).reshape(batch_size, -1)
    beyond_done = torch.full([batch_size], False, dtype=torch.bool)

    s_next, r, d, info = env_model.forward(s_torch, a_torch, beyond_done)

    assert isinstance(s_next, torch.Tensor), "Next state must be a Tensor"
    assert isinstance(r, torch.Tensor), "Reward must be a Tensor"
    assert isinstance(d, torch.Tensor), "Done signal must be a Tensor"
    assert isinstance(info, dict), "Next state must be a Tensor"

    assert s_next.size() == s_torch.size(), "Wrong env dynamics"
    assert (
        len(r.shape) == 1 and r.shape[0] == batch_size
    ), "Wrong reward singal"
    assert (
        len(d.shape) == 1 and d.shape[0] == batch_size
    ), "Wrong done singal"

    if hasattr(env, "constraint_dim") and env.constraint_dim is not None:
        assert (
            "constraint" in info.keys()
        ), "Constraint function must be implemented in info variable"


def check_model(env_name):
    """
    check whether env model class is well defined
    :param env_name: env name
    :return:
    """
    print(f"checking `{env_name}_model` ...")
    try:
        env_cls = check_env_file_structures(env_name + "_data")
        env = env_cls()

    except:
        print(
            f"Can not create `{env_name}`, "
            f"It may because some modules are not installed, "
            f"or The environment is not implemented correctly"
        )
        return None

    env_model_cls = check_env_model_file_structures(env_name + "_model")
    env_model = env_model_cls()

    check_model0(env, env_model)
