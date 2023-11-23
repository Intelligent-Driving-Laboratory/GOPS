import numpy as np

from gops.env.env_gen_ocp.pyth_base import Context, ContextState


class BalancePoint(Context):
    def __init__(self):
        self.state = ContextState(reference=np.zeros(6, dtype=np.float32))

    def reset(self) -> ContextState[np.ndarray]:
        return self.state
    
    def step(self) -> ContextState[np.ndarray]:
        return self.state
    
    def get_zero_state(self) -> ContextState[np.ndarray]:
        return self.state