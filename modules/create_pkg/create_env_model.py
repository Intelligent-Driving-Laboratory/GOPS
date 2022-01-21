#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang ZHANG
#  Description: Create environments
"""
resources:
env = create_env('gym_pendulum_diff')

1: Copy your environment file into env folder, and environment file is named as
    gym_***.py
    gym_***_diff.py
    pyth_***.py
    simu_***.py
2: The environment class is named in camel-case style after file name
    ex: GymMountaincarContiDiff in gym_mountaincar_conti_diff.py
    ex: GymCartpoleConti in gym_cartpole_conti.py
3: Define an instantiating function env_creator() which return a instance of the environment
Note: create_env() requires that either 2 or 3 is satisfied.
"""


#  Update Date: 2020-11-10, Yuhang ZHANG:


def create_env_model(**kwargs):
    env_model_name = kwargs['env_id'] + '_model'
    try:
        file = __import__(env_model_name)
    except NotImplementedError:
        raise NotImplementedError('This environment does not have differential model')

    env_name_camel = formatter(env_model_name)

    if hasattr(file, "env_moedel_creator"):
        y = getattr(file, "env_moedel_creator")
        env_model = y(**kwargs)
    elif hasattr(file, env_name_camel):
        y = getattr(file, env_name_camel)
        env_model = y(**kwargs)
    else:
        raise NotImplementedError("This environment model is not properly defined")
    print("Create environment model successfully!")
    return env_model


def formatter(src: str, firstUpper: bool = True):
    arr = src.split('_')
    res = ''
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
