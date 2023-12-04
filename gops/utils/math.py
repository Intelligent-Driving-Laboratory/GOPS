import math
from typing import Union

import numpy as np
import torch


def angle_normalize(
    x: Union[float, np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray, torch.Tensor]:
    return ((x + math.pi) % (2 * math.pi)) - math.pi
