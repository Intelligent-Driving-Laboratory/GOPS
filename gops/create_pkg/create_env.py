#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code


def create_env(**kwargs):
    env_name = kwargs["env_id"]
    env_name_data = env_name + "_data"
    try:
        file = __import__(env_name_data)
    except NotImplementedError:
        raise NotImplementedError("This environment does not exist")

    env_name_camel = formatter(env_name)

    if hasattr(file, "env_creator"):
        env_class = getattr(file, "env_creator")
        env = env_class(**kwargs)
    elif hasattr(file, env_name_camel):
        env_class = getattr(file, env_name_camel)
        env = env_class(**kwargs)
    else:
        print("Env name: ", env_name_camel)
        raise NotImplementedError("This environment is not properly defined")
    print("Create environment successfully!")
    env.seed(kwargs.get("seed", 111))
    return env


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
