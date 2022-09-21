#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Acrobat Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

from gops.env.resources.simu_doublemass_v2.doublemass import GymEnv
from gops.env.resources.simu_doublemass_v2.doublemass._env import EnvSpec

from gym import spaces
import gym
from gops.env.env_archive.resources import doublemass

def env_creator(**kwargs):
    spec = EnvSpec(
        id="SimuDoubleMassConti-v0",
        max_episode_steps=1000
    )
    return GymEnv(spec)


if __name__ == "__main__":
    from gops.utils.env_utils import random_rollout
    random_rollout(env_creator())
