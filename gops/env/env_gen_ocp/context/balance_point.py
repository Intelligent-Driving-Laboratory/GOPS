from typing import Optional, Sequence

import numpy as np

from gops.env.env_gen_ocp.pyth_base import Context, ContextState


class BalancePoint(Context):
    def __init__(
            self,
            balanced_state: Optional[Sequence] = None,
            index: Optional[Sequence[int]] = None,
    ):
        if index is None:
            index = list(range(len(balanced_state)))
        reference = np.zeros(len(index), dtype=np.float32)
        if balanced_state is not None:
            if not isinstance(balanced_state, np.ndarray):
                balanced_state = np.array(balanced_state, dtype=np.float32)
            reference = balanced_state[index]
        self.state = ContextState(reference=reference)
        self.index = index

    def reset(self) -> ContextState[np.ndarray]:
        return self.state
    
    def step(self) -> ContextState[np.ndarray]:
        return self.state
    
    def get_zero_state(self) -> ContextState[np.ndarray]:
        return self.state
