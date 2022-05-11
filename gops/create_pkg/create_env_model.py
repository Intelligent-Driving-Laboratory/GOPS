#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code


def create_env_model(**kwargs):
    env_model_name = kwargs["env_id"] + "_model"
    try:
        file = __import__(env_model_name)
    except NotImplementedError:
        raise NotImplementedError("This environment does not have differential model")

    env_name_camel = formatter(env_model_name)

    if hasattr(file, "env_model_creator"):
        y = getattr(file, "env_model_creator")
        env_model = y(**kwargs)
    elif hasattr(file, env_name_camel):
        y = getattr(file, env_name_camel)
        env_model = y(**kwargs)
    else:
        raise NotImplementedError("This environment model is not properly defined")
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
