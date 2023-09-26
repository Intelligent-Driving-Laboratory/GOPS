#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: LipsNet
#  Update: 2023-06-14, Xujie Song: create LipsNet function


__all__ = [
    "DetermPolicy",
    "StochaPolicy",
]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import jacrev, vmap

from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution

# A dict supporting different learning_rate for grouped parameters
class Para_dict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @property
    def requires_grad(self):
        return self['params'].requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value):
        self['params'].requires_grad = value

    @property
    def data(self):
        return self['params'].data


# Define K(x)
class Lips_K(nn.Module):
    def __init__(self, local, Lips_start, sizes) -> None:
        super().__init__()
        self.local = local
        if local:
            # declare layers
            layers = []
            for j in range(0, len(sizes) - 2):
                layers += [nn.Linear(sizes[j], sizes[j+1]),
                           nn.Tanh()]
            layers += [nn.Linear(sizes[-2], sizes[-1], bias=True), nn.Softplus()]
            self.K = nn.Sequential(*layers)
            # init weight
            for i in range(len(self.K)):
                if isinstance(self.K[i], nn.Linear):
                    if isinstance(self.K[i+1], nn.ReLU):
                        nn.init.kaiming_normal_(self.K[i].weight, nonlinearity='relu')
                    elif isinstance(self.K[i+1], nn.LeakyReLU):
                        nn.init.kaiming_normal_(self.K[i].weight, nonlinearity='leaky_relu')
                    else:
                        nn.init.xavier_normal_(self.K[i].weight)
            self.K[-2].bias.data += torch.tensor(Lips_start, dtype=torch.float).data
        else:
            self.K = torch.nn.Parameter(torch.tensor(Lips_start, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        if self.local:
            return self.K(x)
        else:
            return F.softplus(self.K).repeat(x.shape[0]).unsqueeze(1)


# Define MLP function through MGN
class LipsNet(nn.Module):
    def __init__(self, sizes, activation, output_activation=nn.Identity,
                 lips_init_value=100, eps=1e-5, lips_auto_adjust=True,
                 loss_lambda=0.1,
                 local_lips=False, lips_hidden_sizes=None) -> None:
        super().__init__()
        # display PyTorch version
        print("Your PyTorch version is", torch.__version__)
        print("To use LipsNet, the PyTorch version must be >=1.12 and <=2.2")

        # declare network
        layers = []
        for j in range(0, len(sizes) - 2):
            layers += [nn.Linear(sizes[j], sizes[j+1]),
                       activation()]
        layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
        self.mlp = nn.Sequential(*layers)
        # init weight
        for i in range(len(self.mlp)):
                if isinstance(self.mlp[i], nn.Linear):
                    if isinstance(self.mlp[i+1], nn.ReLU):
                        nn.init.kaiming_normal_(self.mlp[i].weight, nonlinearity='relu')
                    elif isinstance(self.mlp[i+1], nn.LeakyReLU):
                        nn.init.kaiming_normal_(self.mlp[i].weight, nonlinearity='leaky_relu')
                    else:
                        nn.init.xavier_normal_(self.mlp[i].weight)
        # record the weight is updated or not
        self.para_updated = False
        # local or global lipschitz
        self.local = local_lips
        # declare K(x)
        self.K = Lips_K(local_lips, lips_init_value, lips_hidden_sizes)
        # loss weight
        self.loss_lambda = loss_lambda
        # declare eps in denominator
        self.eps = eps
        # auto_adjust setting
        self.lips_auto_adjust = lips_auto_adjust
        if lips_auto_adjust:
            self.regular_loss = 0
            self.register_full_backward_pre_hook(backward_hook)

    def forward(self, x):
        # calculate K(x)
        K_value = self.K(x)
        # Lipschitz adjustment
        if self.lips_auto_adjust and self.training and K_value.requires_grad:
            # L2 loss
            self.regular_loss += self.loss_lambda * (K_value ** 2).mean()
        
        # forward process
        f_out = self.mlp(x)
        # calcute jac matrix
        if K_value.requires_grad:
            jacobi = vmap(jacrev(self.mlp))(x)
        else:
            with torch.no_grad():
                jacobi = vmap(jacrev(self.mlp))(x)
        # jacobi.dim: (x.shape[0], f_out.shape[1], x.shape[1])
        #             (batch     , f output dim  , x intput dim)
        # calcute jac norm
        norm = torch.norm(jacobi, 2, dim=(1,2)).unsqueeze(1)
        # multi-dimensional gradient normalization (MGN)
        f_out_Lips = K_value * f_out / (norm + self.eps)
        # f_out_Lips = self.K_record * f_out / (norm + f_out.abs())
        return f_out_Lips
    
def backward_hook(module, gout):
    module.regular_loss.backward(retain_graph=True)
    module.regular_loss = 0
    return gout


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]

        lips_init_value = kwargs["lips_init_value"]
        assert lips_init_value is not None

        lips_auto_adjust = kwargs["lips_auto_adjust"]
        assert lips_auto_adjust is not None

        local_lips = kwargs["local_lips"]
        assert local_lips is not None

        lips_hidden_sizes = kwargs["lips_hidden_sizes"]
        if local_lips:
            assert lips_hidden_sizes is not None
            lips_hidden_sizes = [obs_dim] + list(lips_hidden_sizes) + [1]
        
        eps = kwargs.get("eps", 1e-4)

        loss_lambda = kwargs["lambda"]
        assert loss_lambda is not None

        self.squash_action = kwargs["squash_action"]
        assert self.squash_action is not None

        self.learning_rate = kwargs["learning_rate"]
        self.lips_learning_rate = kwargs["lips_learning_rate"]
        assert (self.learning_rate is not None) and (self.lips_learning_rate is not None)

        hidden_sizes = kwargs["hidden_sizes"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        
        self.pi = LipsNet(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
                lips_init_value,
                eps,
                lips_auto_adjust,
                loss_lambda,
                local_lips,
                lips_hidden_sizes
        )

        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.eval()
    
    def parameters(self, recurse: bool = True):
        params = []
        for p in self.pi.mlp.parameters():
            params.append(
                Para_dict(params = p, lr = self.learning_rate)
            )
        for p in self.pi.K.parameters():
            params.append(
                Para_dict(params = p, lr = self.lips_learning_rate)
            )
        return params

    def zero_grad(self, ):
        self.pi.zero_grad()

    def forward(self, obs):
        if self.squash_action:
            action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
                self.pi(obs)
            ) + (self.act_high_lim + self.act_low_lim) / 2
        else:
            action = self.pi(obs)
        return action
    

# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        lips_init_value = kwargs["lips_init_value"]
        assert lips_init_value is not None

        lips_auto_adjust = kwargs["lips_auto_adjust"]
        assert lips_auto_adjust is not None

        local_lips = kwargs["local_lips"]
        assert local_lips is not None

        lips_hidden_sizes = kwargs["lips_hidden_sizes"]
        if local_lips:
            assert lips_hidden_sizes is not None
            lips_hidden_sizes = [obs_dim] + list(lips_hidden_sizes) + [1]
        
        eps = kwargs.get("eps", 1e-4)

        loss_lambda = kwargs["lambda"]
        assert loss_lambda is not None

        self.squash_action = kwargs["squash_action"]
        assert self.squash_action is not None

        self.learning_rate = kwargs["learning_rate"]
        self.lips_learning_rate = kwargs["lips_learning_rate"]
        assert (self.learning_rate is not None) and (self.lips_learning_rate is not None)

        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.pi = LipsNet(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
                lips_init_value,
                eps,
                lips_auto_adjust,
                loss_lambda,
                local_lips,
                lips_hidden_sizes
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.pi = LipsNet(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
                lips_init_value,
                eps,
                lips_auto_adjust,
                loss_lambda,
                local_lips,
                lips_hidden_sizes
            )
            self.log_std = nn.Parameter(torch.zeros(1, act_dim)) # not used
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.pi = LipsNet(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
                lips_init_value,
                eps,
                lips_auto_adjust,
                loss_lambda,
                local_lips,
                lips_hidden_sizes
            )
            self.log_std = nn.Parameter(torch.zeros(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        self.eval()

    def parameters(self, recurse: bool = True):
        params = []
        for p in self.pi.mlp.parameters():
            params.append(
                Para_dict(params = p, lr = self.learning_rate)
            )
        for p in self.pi.K.parameters():
            params.append(
                Para_dict(params = p, lr = self.lips_learning_rate)
            )
        return params
    
    def zero_grad(self, ):
        self.pi.zero_grad()
        self.log_std.zero_grad()

    def forward(self, obs):
        if self.std_type == "mlp_separated":
            # action mean
            if self.squash_action:
                action_mean = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
                    self.pi(obs)
                ) + (self.act_high_lim + self.act_low_lim) / 2
            else:
                action_mean = self.pi(obs)
            # action std
            action_std = torch.clamp(
                self.log_std(obs), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.pi(obs)
            action_mean, action_log_std = torch.chunk(logits, chunks=2, dim=-1)  # output the mean
            # action mean
            if self.squash_action:
                action_mean = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
                    action_mean
                ) + (self.act_high_lim + self.act_low_lim) / 2
            # action std
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            # action mean
            if self.squash_action:
                action_mean = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
                    self.pi(obs)
                ) + (self.act_high_lim + self.act_low_lim) / 2
            else:
                action_mean = self.pi(obs)
            # action std
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


if __name__ == "__main__":
    net = LipsNet(sizes=[2,2], activation=nn.ReLU)

    print(net.parameters())

    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())
    
    exit()