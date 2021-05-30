from gym import spaces
import gym
from modules.env.resources import doublemass
import numpy as np

class SimuDoublemassconti(gym.Env):

    def __init__(self):
        self._physics = doublemass.model_wrapper()
        self.action_space = spaces.Box(low=np.array(self._physics.get_param()['a_min']).reshape(-1), high=np.array(self._physics.get_param()['a_max']).reshape(-1))
        self.observation_space = spaces.Box(np.array(self._physics.get_param()['x_min'].reshape(-1)), np.array(self._physics.get_param()['x_max']).reshape(-1))
        self.reset()

    def step(self, action):
        state, is_done, reward = self._step_physics({'Action': action})
        self.cstep += 1
        is_done += self.cstep>2000
        return state, reward, is_done, {}

    def reset(self):
        self._physics.terminate()
        self._physics = doublemass.model_wrapper()

        # randomized initiate
        state = np.random.uniform(low=[-1,-0.5,1,-0.5], high=[1, 0.5, 2, 0.5], size=(4,))
        param = self._physics.get_param()
        param.update(list(zip(('x_ini'), state.tolist())))
        self._physics.set_param(param)
        self._physics.initialize()
        self.cstep = 0
        return state

    def render(self):
        pass

    def close(self):
        self._physics.renderterminate()

    def _step_physics(self, action):
        return self._physics.step(action)


if __name__ == "__main__":
    import gym
    import numpy as np

    env = SimuDoublemassconti()
    s = env.reset()
    for i in range(50):
        a = np.ones([1])*20
        sp, r, d, _ = env.step(a)
        print(s, a, r, d)
        s = sp
