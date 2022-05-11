from abc import ABCMeta, ABC, abstractmethod

from typing import Tuple, Type

from gops.utils.utils import set_seed
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.utils import get_apprfunc_dict
import warnings
import torch


class ApprBase(ABC, torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs["cnn_shared"]:
            feature_args = get_apprfunc_dict("feature", kwargs["value_func_type"], **kwargs)
            kwargs["feature_net"] = create_apprfunc(**feature_args)


class AlgorithmBase(metaclass=ABCMeta):
    def __init__(self, index, **kwargs):
        self.networks = None
        set_seed(kwargs["trainer"], kwargs["seed"], index + 300)

    @property
    @abstractmethod
    def adjustable_parameters(self) -> tuple:
        pass

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key) and key in self.adjustable_parameters:
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not adjustable in algorithm!"
                warnings.warn(warning_msg)

    def get_parameters(self):
        params = dict(zip(self.adjustable_parameters,
                          (getattr(self, para) for para in self.adjustable_parameters)))
        return params

    def state_dict(self):
        return self.networks.state_dict()

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    @abstractmethod
    def local_update(self, data: dict, iteration: int) -> dict:
        pass

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        raise NotImplemented

    def remote_update(self, update_info: dict):
        raise NotImplemented

    def to(self, device):
        self.networks.to(device)
