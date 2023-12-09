import numpy as np
from gops.env.env_gen_ocp.pyth_base import Robot
from scipy.linalg._solvers import solve_discrete_are


class LqModel(Robot):
    def __init__(self, config) -> None:
        super(LqModel).__init__()
        self.A = np.array(config["A"], dtype=np.float32)
        self.B = np.array(config["B"], dtype=np.float32)
        self.Q = np.array(config["Q"], dtype=np.float32)
        self.R = np.array(config["R"], dtype=np.float32)

        self.time_step = config["dt"]
        self.K, self.P = self.compute_control_matrix()

        self.reward_scale = config["reward_scale"]
        self.reward_shift = config["reward_shift"]
        self.state_dim = self.A.shape[0]

        # IA = (1 - A * dt)
        self.IA = np.eye(self.state_dim) - self.A * self.time_step
        self.inv_IA = np.linalg.pinv(self.IA)

        self.state = None

    def step(self, action: np.ndarray) -> np.ndarray:
        self.state = self.prediction(self.state, action)
        return self.state

    def prediction(self, x_t, u_t):
        x_t = np.expand_dims(x_t, axis=0)
        u_t = np.expand_dims(u_t, axis=0)

        tmp = np.dot(self.B, u_t.T) * self.time_step + x_t.T
        x_next = np.dot(self.inv_IA, tmp).T
        x_next = x_next.squeeze(0)

        return x_next

    def compute_control_matrix(self):
        gamma = 0.99
        A0 = self.A.astype("float64")
        A = np.linalg.pinv(np.eye(A0.shape[0]) - A0 * self.time_step) * np.sqrt(gamma)
        B0 = self.B.astype("float64")
        B = A @ B0 * self.time_step
        Q = np.diag(self.Q).astype("float64")
        R = np.diag(self.R).astype("float64")
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
        return K, P
