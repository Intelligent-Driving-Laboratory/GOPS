#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: base class for algorithms
#  Update: 2022-12-03, Wenxuan Wang: create bass class for algorithms
#  Update: 2023-08-28, Guojian Zhan: support lr schedule


from abc import ABCMeta, ABC, abstractmethod

from typing import Tuple, Type

from gops.utils.common_utils import set_seed
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.common_utils import get_apprfunc_dict
import torch


class ApprBase(ABC, torch.nn.Module):
    """Base Class of Approximate function container"""

    def __init__(self, **kwargs):
        super().__init__()
        # Create a shared feature networks for value function and policy function
        if kwargs["cnn_shared"]:
            feature_args = get_apprfunc_dict("feature", **kwargs)
            kwargs["feature_net"] = create_apprfunc(**feature_args)

    def init_scheduler(self, **kwargs):
        # self.optimizer_dict should be initialized in alg before calling this function
        assert hasattr(self, "optimizer_dict")
        self.scheduler_dict = {}
        scheduler_keys = [key for key in kwargs if key.endswith("_scheduler")]
        scheduler_args = {
            key: kwargs[key] for key in scheduler_keys
        }
        for key in scheduler_keys:
            self.scheduler_dict[key] = getattr(
                    torch.optim.lr_scheduler,
                    scheduler_args[key]["name"],
                )(
                    self.optimizer_dict[key.replace("_scheduler", "")],
                    **scheduler_args[key]["params"],
                )

class AlgorithmBase(metaclass=ABCMeta):
    """Base Class of Algorithm

    Args:
        int     index       : used for calculating offset of random seed for subprocess.
    """
    

    def __init__(self, index, **kwargs):
        self.networks = None
        set_seed(kwargs["trainer"], kwargs["seed"], index + 300)

    @property
    @abstractmethod
    def adjustable_parameters(self) -> tuple:
        """Return all the adjustable hyperparameters of the algorithm"""
        pass

    def set_parameters(self, param_dict):
        """Set hyperparameters of the algorithm"""
        for key in param_dict:
            if hasattr(self, key) and key in self.adjustable_parameters:
                setattr(self, key, param_dict[key])
            else:
                error_msg = "param '" + key + "'is not adjustable in algorithm!"
                raise RuntimeError(error_msg)

    def get_parameters(self):
        """Get the current hyperparameters of the algorithm"""
        params = dict(
            zip(
                self.adjustable_parameters,
                (getattr(self, para) for para in self.adjustable_parameters),
            )
        )
        return params

    def state_dict(self):
        return self.networks.state_dict()

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def local_update(self, data: dict, iteration: int) -> dict:
        tb_info = self._local_update(data, iteration)
        for key, scheduler in self.networks.scheduler_dict.items():
            scheduler.step()
        return tb_info

    def remote_update(self, update_info: dict):
        self._remote_update(update_info)
        for key, scheduler in self.networks.scheduler_dict.items():
            scheduler.step()

    def _local_update(self, data: dict, iteration: int) -> dict:
        pass

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        raise NotImplemented

    def _remote_update(self, update_info: dict):
        raise NotImplemented

    def to(self, device):
        self.networks.to(device)

    def train(self):
        self.networks.train()

    def eval(self):
        self.networks.eval()