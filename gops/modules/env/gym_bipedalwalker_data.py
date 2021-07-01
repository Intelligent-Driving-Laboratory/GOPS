#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Bipedialwalker Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator(**kwargs):
    try:
        return gym.make('BipedalWalker-v3')
    except AttributeError:
        raise ModuleNotFoundError("Warning: Box2d is not installed")

