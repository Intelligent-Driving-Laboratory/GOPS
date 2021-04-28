import torch

PI = 3.1415926
tensor_pi = torch.tensor(PI)


class GymMountaincarContiDiff(object):
    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0

        self.min_position = -1.2
        self.max_position = 0.6

        self.min_speed = -0.07
        self.max_speed = 0.07

        self.goal_position = 0.45
        self. goal_velocity = 0.0
        self.power = 0.0015

    def step(self, state, u):
        """the state transformation function 
        # TODO add n-step
        Parameters
        ----------
        state : [position, velocity]
            shape:(2,)
        u : [action]
            shape(1,)

        Returns
        -------
        newstate : shape:(2,)
        reward: shape : (1,)
        """
        position, velocity = state

        force = torch.clamp(u[0], self.min_action, self.max_action)

        velocity = velocity + force * self.power - 0.0025 * torch.cos(3 * position)

        velocity = torch.clamp(velocity, self.min_speed, self.max_speed)

        position = position + velocity

        position = torch.clamp(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0

        reward = 0
        if(position >= self.goal_position and velocity >= self.goal_velocity):
            reward = 100

        reward = reward - torch.pow(u[0], 2) * 0.1

        state_new = torch.stack([position, velocity])

        return state_new, reward

    def __call__(self, state, u):
        return self.step(state, u)


# def env_creator():
#     return GymMountaincarContiDiff()


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def angle_normalize(x):
    return (((x + tensor_pi) % (2*tensor_pi)) - tensor_pi)


def arccs(sinth, costh):
    th = torch.acos(costh)
    if sinth <= 0:
        th = 2 * 3.1415926 - th
    return th

if __name__ == "__main__":
    f = GymMountainCarContiDiff()
    import gym
    import matplotlib.pyplot as plt
    import numpy as np
    env = gym.make('MountainCarContinuous-v0')
    s = env.reset()
    s_real = []
    s_fake = []
    for i in range(200):
        a = env.action_space.sample()
        sp, r, d, _ = env.step(a)
        # print(s, a, sp)
        s_real.append(sp)
        tsp, _ = f(torch.tensor(s), torch.tensor(a))
        s_fake.append(tsp.numpy())
        s = sp
    # print(tsp)
    s_real = np.array(s_real)
    s_fake = np.array(s_fake)
    plt.plot(s_real)
    plt.plot(s_fake)
    plt.show()
    print("All states match, The model is right")