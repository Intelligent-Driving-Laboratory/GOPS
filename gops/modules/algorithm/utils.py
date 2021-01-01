"""


"""

import torch
import torch.nn as nn
import parser


def get_apprfunc_dict(name, **kwargs):
    pass


class ActorCriticApprFunc(nn.Module):
    def __init__(self, actor,critic):
        super().__init__()
        self.pi = actor
        self.q = critic

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
