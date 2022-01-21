#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Breakout Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator(**kwargs):
    try:
        return gym.make('Breakout-v0')
    except:
        raise ModuleNotFoundError('Warning: Atari_py is not installed')
