#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Car-racing Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym
import numpy as np
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.transform_observation import TransformObservation
from gym.spaces import Box


def env_creator(**kwargs):
    try:
        env_obj = gym.make('CarRacing-v0')
        env_obj = GrayScaleObservation(env_obj)
        env_obj = FrameStack(env_obj, 4)
        f = lambda x: np.transpose(x, [1, 2, 0])
        env_obj = TransformObservation(env_obj, f)
        env_obj.observation_space = Box(low=0, high=255, shape=(96, 96, 4), dtype=np.uint8)
        return env_obj
    except AttributeError:
        raise ModuleNotFoundError("Warning: Box2d or Swig are not installed")


# e = env_creator()
#
# s = e.reset()
# print(type(s))
# for i in range(1):
#     s, r, d, _ = e.step(e.action_space.sample())
#     e.render()
#     print(type(s))