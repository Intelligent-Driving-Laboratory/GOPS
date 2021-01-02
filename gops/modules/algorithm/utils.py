"""


"""

import torch
import torch.nn as nn
import parser


def get_apprfunc_dict(name, **kwargs):
    paras = {}
    return paras


class ActorCriticApprFunc(nn.Module):
    def __init__(self, pi,q):
        super().__init__()
        self.pi = pi
        self.q = q

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
