#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Hottercolder Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator():
    return gym.make('HotterColder-v0')



if __name__ == '__main__':
    env = env_creator()

    env.reset()
    for i in range(100):
        a = env.action_space.sample()
        s, r, d, _ = env.step(a)
        print('s', s)
        print('a', a)
        print('r', r)
        print('d', d)