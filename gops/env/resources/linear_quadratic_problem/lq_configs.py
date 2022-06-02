import warnings

import numpy as np

config_s3a1 = dict(
    A=[
        [-1.01887, 0.90506, -0.00215],
        [0.82225, -1.07741, -0.17555],
        [0.0, 0.0, -1.0000]
    ],
    B=[
        [0.0],
        [0.0],
        [5.0]
    ],
    Q=[50.0, 0.0, 0.0],
    R=[1.0],
    dt=0.05,
    init_mean=[0, 0, 0],
    init_std=[0.1, 0.2, 0.2],
    state_high=[np.inf, np.inf, np.inf],
    state_low=[-np.inf, -np.inf, -np.inf],
    action_high=[1.0],
    action_low=[-1.0],
    max_step=200,
)


Izz = 1536.7
vx = 20
k1 = 72200
k2 = 39912
la = 1.015
lb = 1.895
m = 1270

config_s4a1 = dict(
    A=[
        [-(k1 + k2) / m / vx, -(la * k1 - lb * k2) / m / vx - vx, 0, 0],
        [-(la * k1 - lb * k2) / Izz / vx, -(la * la * k1 - lb * lb * k2) / Izz / vx, 0, 0],
        [1, 0.0, 0, vx],
        [0, 1, 0, 0]
    ],
    B=[
        [k1 / m],
        [la * k1 / Izz],
        [0.0],
        [0.0]
    ],
    Q=[1.0, 1.0, 1.0, 1.0],
    R=[1.0],
    dt=0.05,
    init_mean=[0, 0, 0, 0],
    init_std=[0.1, 0.1, 0.1, 0.1],
    state_high=[np.inf, np.inf, np.inf, np.inf],
    state_low=[-np.inf, -np.inf, -np.inf, -np.inf],
    action_high=[1.0],
    action_low=[-1.0],
    max_step=200,
)

config_s5a1 = dict(
    A=[
        [-0.2, 0.5, 0, 0, 0],
        [0, -0.5, 1.6, 0, 0],
        [0, 0.0, -14.3, 95.9, 0],
        [0, 0, 0, -33.3, -100],
        [0, 0, 0, 0, 0]
    ],
    B=[
        [0],
        [0],
        [0.0],
        [0.0],
        [30]
    ],
    Q=[1.0, 0.0, 0.0, 0.0, 0.0],
    R=[1.0],
    dt=0.05,
    init_mean=[0, 0, 0, 0, 0],
    init_std=[0.1, 0.1, 0.1, 0.1, 0.1],
    state_high=[np.inf, np.inf, np.inf, np.inf, np.inf],
    state_low=[-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    action_high=[1.0],
    action_low=[-1.0],
    max_step=200,
)

config_s4a2 = dict(
    A=[
        [0, 1, 0, 0],
        [-76.6, 0, 0, 0],
        [0, 0.0, 0, 1],
        [-0.09, 0, 0, 0]
    ],
    B=[
        [0, 0],
        [-10.22, -10.22],
        [0.0, 0],
        [1.37, 1.37]
    ],
    Q=[200, 200, 50, 50],
    R=[1.0, 1.0],
    dt=0.05,
    init_mean=[0, 0, 0, 0],
    init_std=[0.1, 0.1, 0.1, 0.1],
    state_high=[np.inf, np.inf, np.inf, np.inf],
    state_low=[-np.inf, -np.inf, -np.inf, -np.inf],
    action_high=[1.0, 1.0],
    action_low=[-1.0, -1.0],
    max_step=200,
)

config_s6a3 = dict(
    A=[
        [0, 1, 0, 0, 0, 0],
        [30, 0, 0, 0, 0, 0],
        [0, 0.0, 0, 1, 0, 0],
        [25.5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [-20, 0, 0, 0, 0, 0]
    ],
    B=[
        [0, 0, 0],
        [10.5, 10.5, 10.5],
        [0.0, 0, 0],
        [1.5, 1.5, 1.5],
        [0, 0, 0],
        [23, 23, 23]
    ],
    Q=[0, 20, 0, 20, 0, 50],
    R=[1.0, 1.0, 1],
    dt=0.05,
    init_mean=[0, 0, 0, 0,  0, 0],
    init_std=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    state_high=[np.inf, np.inf, np.inf, np.inf , np.inf, np.inf],
    state_low=[-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    action_high=[1.0, 1.0, 1.0],
    action_low=[-1.0, -1.0, -1.0],
    max_step=200,
)


def _check_1d_vector(vec, num):
    vec = np.array(vec)
    assert len(vec.shape) == 1
    assert vec.shape[0] == num


def controllability(A, B):
    AB={}
    AB[0] = B
    for i in range(1, A.shape[0]):
        AB[i] = A.dot(AB[i-1])
    q = np.column_stack((AB.values()))

    z = q.dot(q.T)
    if np.linalg.matrix_rank(z) != A.shape[0]:
        print(" >>> The system is uncontrollable, s{}a{}".format(*B.shape))


def check_lq_config(cfg):
    A = np.array(cfg["A"])
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]

    state_dim = A.shape[0]
    B = np.array(cfg["B"])
    assert len(B.shape) == 2
    assert B.shape[0] == state_dim
    action_dim = B.shape[1]

    _check_1d_vector(cfg["Q"], state_dim)

    _check_1d_vector(cfg["R"], action_dim)

    _check_1d_vector(cfg["init_mean"], state_dim)

    _check_1d_vector(cfg["init_std"], state_dim)

    _check_1d_vector(cfg["state_high"], state_dim)

    _check_1d_vector(cfg["state_low"], state_dim)

    _check_1d_vector(cfg["action_high"], action_dim)

    _check_1d_vector(cfg["action_low"], action_dim)

    assert cfg["max_step"] >= 0
    assert cfg["dt"] >= 0

    controllability(A, B)


def test_all_configs():
    check_lq_config(config_s3a1)

    check_lq_config(config_s4a1)

    check_lq_config(config_s5a1)

    check_lq_config(config_s4a2)

    check_lq_config(config_s6a3)


if __name__ == "__main__":
    test_all_configs()

