from gops.env.resources.crossing import endtoend



class GymCrossingData():
    def __init__(self):
        self._env = endtoend.CrossroadEnd2endMixPiFix('left')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.adv_action_space = None

    def reset(self):
        s = self._env.reset()
        return s

    def step(self, action, adv_action=None):
        return self._env.step(action)

    def get_constraints(self):
        return self._env.get_constraints()

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        self._env.close()




if __name__ == '__main__':
    env = GymCrossingData()
    env.reset()
    for _ in range(100):
        a = env.action_space.sample()
        env.step(a)

        c = env.get_constraints()
        print(c.shape)
        env.render()
    env.close()