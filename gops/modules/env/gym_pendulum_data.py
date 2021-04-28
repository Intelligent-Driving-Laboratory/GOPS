import gym


def env_creator():
    return gym.make("Pendulum-v0")