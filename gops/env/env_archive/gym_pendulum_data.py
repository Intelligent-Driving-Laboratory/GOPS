#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Pendulum Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment


import gym
from gops.utils.env_utils import safe_make


def env_creator(**kwargs):
    return safe_make("Pendulum-v0")


if __name__ == "__main__":
    env = env_creator()
