#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: lunarlander Environment (continous version)
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator():
    try:
        return gym.make('LunarLanderContinuous-v2')
    except AttributeError:
        raise ModuleNotFoundError("Box2d is not installed")

