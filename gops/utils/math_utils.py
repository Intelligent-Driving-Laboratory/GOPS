import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn


def angle_normalize(
    x: Union[float, np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray, torch.Tensor]:
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def incremental_update(net_from: nn.Module, net_to: nn.Module, tau: float):
    for (p, p_tar) in zip(net_from.parameters(), net_to.parameters()):
        p_tar.data.mul_(1 - tau).add_(tau * p.data)
