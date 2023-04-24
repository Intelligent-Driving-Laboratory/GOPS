#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Simulink Aircraft Environment
#  Update: 2021-05-05, Yuxuan Jiang: create environment

from gops.env.env_matlab.resources.simu_aircraft_v2.aircraft import GymEnv
from gops.env.env_matlab.resources.simu_aircraft_v2.aircraft._env import EnvSpec


def env_creator(**kwargs):
    spec = EnvSpec(id="SimuAircraftConti-v0", max_episode_steps=200)
    return GymEnv(spec)
