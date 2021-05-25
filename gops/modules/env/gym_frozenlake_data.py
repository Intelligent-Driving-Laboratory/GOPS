#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Frozenlake Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator():
    return gym.make('FrozenLake-v0')
