import gym


def env_creator():
    return gym.make('Acrobot-v1')


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