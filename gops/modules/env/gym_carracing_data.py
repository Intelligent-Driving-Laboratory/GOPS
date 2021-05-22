#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Car-racing Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator():
    try:
        return gym.make('CarRacing-v0')
    except AttributeError:
        raise ModuleNotFoundError("Warning: Box2d or Swig are not installed")
