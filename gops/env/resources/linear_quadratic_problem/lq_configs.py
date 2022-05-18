import numpy as np

config_s3a1 = dict(
    A=[[-1.01887, 0.90506, -0.00215], [0.82225, -1.07741, -0.17555], [0.0, 0.0, -1.0000]],
    B=[[0.0], [0.0], [5.0]],
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

