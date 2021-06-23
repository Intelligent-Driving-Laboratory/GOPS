#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Car-racing Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation


def env_creator(**kwargs):
    try:
        env_obj = gym.make('CarRacing-v0')
        env_obj = GrayScaleObservation(env_obj)
        env_obj = FrameStack(env_obj, 4)
        return env_obj
    except AttributeError:
        raise ModuleNotFoundError("Warning: Box2d or Swig are not installed")


# e = env_creator()
#
# s = e.reset()
#
# for i in range(100):
#     s, r, d, _ = e.step(e.action_space.sample())
#     e.render()
#     print(s.shape)