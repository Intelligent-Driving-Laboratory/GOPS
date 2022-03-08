#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description:
#  Update: 2021.03.05, Shengbo LI (example, can be deleted)



import re
import gym
from gym import error
env_id_re = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")

def safe_make(env_id):
    registry = gym.envs.registration.registry
    match = env_id_re.search(env_id)
    env_name = match.group(1)
    matching_envs = [
        valid_env_name
        for valid_env_name, valid_env_spec in registry.env_specs.items()
        if env_name == valid_env_spec._env_name
    ]
    if env_id in matching_envs:
        return gym.make(env_id)
    else:
        if len(matching_envs) == 0:
            raise error.UnregisteredEnv("No registered env with id: {}".format(env_id))
        else:
            valid_env_id = matching_envs[-1]
            print("Wrong env version, return a valid version: {} instead".format(valid_env_id))
            return gym.make(valid_env_id)