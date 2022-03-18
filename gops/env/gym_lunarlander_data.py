#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: lunarlander Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment



import gym


def env_creator(**kwargs):
    try:
        return gym.make('LunarLander-v2')
    except AttributeError:
        raise ModuleNotFoundError("Box2d is not installed")


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