#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Simulink cartpole environment
#  Update Date: 2021-07-011, Wenxuan Wang: create simulink environment

from gops.env.resources.simu_cartpole_v2.cartpole import GymEnv
from gops.env.resources.simu_cartpole_v2.cartpole._env import EnvSpec

from gym import spaces
import gym
from gops.env.env_archive.resources import cartpole

def env_creator(**kwargs):
    spec = EnvSpec(
        id="SimuCartPoleConti-v0",
        max_episode_steps=200
    )
    return GymEnv(spec)


if __name__ == "__main__":
    from gops.utils.env_utils import random_rollout
    random_rollout(env_creator())
