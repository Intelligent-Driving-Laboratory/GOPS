#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Pendulum Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator():
    return gym.make("Pendulum-v0")