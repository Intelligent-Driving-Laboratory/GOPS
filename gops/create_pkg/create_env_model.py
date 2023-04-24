#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code

from gops.env.wrapper.wrapping_utils import wrapping_model


def create_env_model(**kwargs):
    env_model_name = kwargs["env_id"] + "_model"
    try:
        file = __import__(env_model_name)
    except NotImplementedError:
        raise NotImplementedError("This environment does not have differential model")

    if "device" not in kwargs.keys():
        if kwargs.get("use_gpu", False):
            kwargs["device"] = "cuda"
        else:
            kwargs["device"] = "cpu"

    env_name_camel = formatter(env_model_name)

    if hasattr(file, "env_model_creator"):
        env_model_class = getattr(file, "env_model_creator")
        env_model = env_model_class(**kwargs)
    elif hasattr(file, env_name_camel):
        env_model_class = getattr(file, env_name_camel)
        env_model = env_model_class(**kwargs)
    else:
        raise NotImplementedError("This environment model is not properly defined")

    reward_scale = kwargs.get("reward_scale", None)
    reward_shift = kwargs.get("reward_shift", None)
    obs_scale = kwargs.get("obs_scale", None)
    obs_shift = kwargs.get("obs_shift", None)
    clip_obs = kwargs.get("clip_obs", True)
    clip_action = kwargs.get("clip_action", True)
    mask_at_done = kwargs.get("mask_at_done", True)
    env_model = wrapping_model(
        model=env_model,
        reward_shift=reward_shift,
        reward_scale=reward_scale,
        obs_shift=obs_shift,
        obs_scale=obs_scale,
        clip_obs=clip_obs,
        clip_action=clip_action,
        mask_at_done=mask_at_done,
    )
    # Print("wrap_model with", reward_shift, reward_scale, obs_shift, obs_scale)
    print("Create environment model successfully!")
    return env_model


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
