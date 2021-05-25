#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Acrobat Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import math
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import torch


class GymCartpolecontiModel:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0
        self.Q_mat = torch.from_numpy(np.diagflat([0.1,0.1,100,0.1])).to(dtype=torch.float32)
        self.R_mat = torch.from_numpy(np.diagflat([0.0001])).to(dtype=torch.float32)
        self.overpenalty = 500

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  #12deg
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds

        self.max_x = self.x_threshold * 2
        self.min_x = -self.max_x

        self.max_x_dot = torch.finfo(torch.float32).max
        self.min_x_dot = -self.max_x_dot

        self.max_theta = self.theta_threshold_radians * 2  # 24deg
        self.min_theta = -self.max_theta

        self.max_theta_dot = torch.finfo(torch.float32).max
        self.min_theta_dot = -self.max_theta_dot
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high)

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _physics(self, state, force):
        x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (torch.squeeze(force) + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        #print(theta)
        new_state = torch.stack([x, x_dot, theta, theta_dot]).transpose(1, 0)
        return new_state



    def step(self, state, action):
        """the state transformation function
        Parameters
        ----------
        state :
            shape:(batch, 4)
        u : [action]
            shape(batch, 1)

        Returns
        -------
        newstate : shape:(batch, 4)
        reward: shape : (batch, 1)
        """
        #action = torch.unsqueeze(action, 0)
        force = self.force_mag * action
        #force.requires_grad_()
        new_state = self._physics(state, force)

        '''derivative_fu = torch.autograd.grad(new_state,force)'''
        x, x_dot, theta, theta_dot = new_state[:, 0], new_state[:, 1], new_state[:, 2], new_state[:, 3]

        done = (x < -self.x_threshold) + \
               (x > self.x_threshold) + \
               (theta < -self.theta_threshold_radians) + \
               (theta > self.theta_threshold_radians)
        done = torch.unsqueeze(done, 1)
        new_state = (~done)*new_state+done*state

        ''' reward = torch.mul(torch.mm(state,self.Q_mat),state).sum(dim=1,keepdim=False) \
                 + torch.mul(torch.mm(force,self.R_mat),force).sum(dim=1,keepdim=False)+\
                 self.overpenalty*done.float()
             reward = torch.unsqueeze(reward,1)
        '''


        reward = 1 - done.float()
        return new_state, reward, done, {}

    def __call__(self, state, action):
        return self.step(state, action)

if __name__ == "__main__":
    f = GymCartpoleContiModel()
    from modules.env.gym_cartpoleconti_data import GymCartpoleConti
    import matplotlib.pyplot as plt
    import numpy as np
    env = GymCartpoleConti()
    s = env.reset()
    s=s.astype(np.float32)
    s_real = []
    s_fake = []
    a = env.action_space.sample()
    a = torch.from_numpy(a)
    a.requires_grad_()
    a.retain_grad()
    tsp, _, done, _=f(torch.tensor(s).view([1, 4]), torch.unsqueeze(a,0))
    tsp[0][1].backward(retain_graph= True)
    print(a.grad)

'''
    for i in range(200):
        #print(i)
        a = env.action_space.sample()
        sp, r, d, _ = env.step(a)
        # print(s, a, sp)
        sp = sp.astype(np.float32)
        s_real.append(sp)
        # print(tts.shape)
        tsp, _, done, _ = f(torch.tensor(s).view([1, 4]), torch.tensor(a).view([1, 1]))
       # print(tsp.shape)
        s_fake.append(tsp.detach().numpy().astype(np.float32))
        if done:
            print(i)
            print(s)
            break
        s = sp

    # print(tsp)
    s_real = np.array(s_real)
    s_fake = np.hstack(s_fake)
    s_fake = s_fake.reshape(-1,4)
    plt.plot(s_real)
    plt.show()
    plt.plot(s_fake)
    plt.show()
    print("All states match, The model is right")
    s = torch.zeros([10, 4])
    a = torch.zeros([10, 1])
    sp = f(s, a)
    print(sp)
    print("batch_support") 
'''