"""
Author: SUN-Hao


u0[0]: ax  m / s ^ 2，
u0[1]: front wheel angle  rad。

stata[0]: x
state[1]: y
state[2]: vx
state[3]: vy
state[4]: theta
state[5]: omega

"""


import numpy as np
import torch
import matplotlib.pyplot as plt

a=1.463
b=1.585
m=1818.2
Iz=3885
kf=-62618
kr=-110185


def dynamic_data(x0 ,u0 ,T):
    x1 = np.zeros(len(x0))

    x1[0] = x0[0] + T* (x0[2] * np.cos(x0[4]) - x0[3] * np.sin(x0[4]))
    x1[1] = x0[1] + T * (x0[3] * np.cos(x0[4]) + x0[2] * np.sin(x0[4]))
    x1[2] = x0[2] + T * u0[0]
    x1[3] = (-(a * kf - b * kr) * x0[5] + kf * u0[1] * x0[2] + m * x0[5] * x0[2] * x0[2] - m * x0[2] * x0[3] / T) / (
                kf + kr - m * x0[2] / T)
    x1[4] = x0[4] + T * x0[5]
    x1[5] = (-Iz * x0[5] * x0[2] / T - (a * kf - b * kr) * x0[3] + a * kf * u0[1] * x0[2]) / (
                (a * a * kf + b * b * kr) - Iz * x0[2] / T)

    return x1


def dynamic_model(x0, u0, T):
    x1 = torch.zeros(len(x0))

    x1[0] = x0[0] + T * (x0[2] * torch.cos(x0[4]) - x0[3] * torch.sin(x0[4])) # x
    x1[1] = x0[1] + T * (x0[3] * torch.cos(x0[4]) + x0[2] * torch.sin(x0[4]))# y
    x1[2] = x0[2] + T * u0[0] # vx
    x1[3] = (-(a * kf - b * kr) * x0[5] + kf * u0[1] * x0[2] + m * x0[5] * x0[2] * x0[2] - m * x0[2] * x0[3] / T) / (
                kf + kr - m * x0[2] / T)
    x1[4] = x0[4] + T * x0[5]
    x1[5] = (-Iz * x0[5] * x0[2] / T - (a * kf - b * kr) * x0[3] + a * kf * u0[1] * x0[2]) / (
                (a * a * kf + b * b * kr) - Iz * x0[2] / T)

    return x1


if __name__=="__main__":
    state = np.zeros(6)
    state[2] = 10
    u = np.zeros(2)
    u[1] = -0.05

    x= list()
    y = list()
    vx = list()
    vy = list()
    omega = list()
    theta = list()

    x.append(state[0])
    y.append(state[1])
    vx.append(state[2])
    vy.append(state[3])
    theta.append(state[4])
    omega.append(state[5])

    for i in range(1000):
        state = dynamic_data(x0=state,u0=u,T=0.01)
        x.append(state[0])
        y.append(state[1])
        vx.append(state[2])
        vy.append(state[3])
        theta.append(state[4])
        omega.append(state[5])

    plt.plot(x,y)
    plt.show()
    plt.plot(theta)
    plt.show()
    plt.plot(vy)
    plt.show()
    plt.plot(omega)
    plt.show()


