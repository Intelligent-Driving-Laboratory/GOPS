import gym


def env_creator():
    try:
        return gym.make('CarRacing-v0')
    except AttributeError:
        raise ModuleNotFoundError("Box2d and Swig not install")


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