#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Acrobat Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

from gops.env.env_matlab.resources.simu_doublemass_v2.doublemass import GymEnv
from gops.env.env_matlab.resources.simu_doublemass_v2.doublemass._env import EnvSpec

from gym import spaces
import gym

def env_creator(**kwargs):
    spec = EnvSpec(
        id="SimuDoubleMassConti-v0",
        max_episode_steps=1000
    )
    return GymEnv(spec)

