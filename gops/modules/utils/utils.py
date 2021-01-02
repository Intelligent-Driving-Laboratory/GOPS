"""


"""

import torch
import torch.nn as nn


def get_activation_func(key : str):
    assert isinstance(key,str)

    activation_func = None
    if key == 'relu':
        activation_func = nn.ReLU
    elif key== 'tanh':
        activation_func = nn.Tanh

    if activation_func is None:
        print('input activation name:' + key)
        raise RuntimeError

    return activation_func


def get_apprfunc_dict(key : str, **kwargs):
    var = {'apprfunc': kwargs[key+'_func_type'],
           'name': kwargs[key+'_func_name'],
           'hidden_sizes':kwargs[key+'_hidden_sizes'],
           'hidden_activation': kwargs[key+'_hidden_activation'],
           'output_activation': kwargs[key+'_output_activation'],
           'obs_dim': kwargs['obsv_dim'],
           'act_dim': kwargs['action_dim'],
           'action_high_limit':kwargs['action_high_limit']
           }
    return var


class ActorCriticApprFunc(nn.Module):
    def __init__(self, pi,q):
        super().__init__()
        self.pi = pi
        self.q = q

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
