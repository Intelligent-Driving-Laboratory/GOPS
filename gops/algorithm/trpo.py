#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Yuxuan JIANG
#  Description: Trust region policy optimization


__all__ = ["TRPO"]

from copy import deepcopy
from typing import Callable, Dict, List
import time

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.autograd

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.action_distributions import GaussDistribution, CategoricalDistribution
from gops.utils.utils import get_apprfunc_dict
from gops.utils.tensorboard_tools import tb_tags


EPSILON = 1e-8

class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        value_func_type = kwargs['value_func_type']
        v_args = get_apprfunc_dict('value', value_func_type, **kwargs)
        self.v: nn.Module = create_apprfunc(**v_args)
        self.v_optimizer = Adam(self.v.parameters(), lr=kwargs['value_learning_rate'])
        
        policy_func_type = kwargs['policy_func_type']
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)

    def update(self, grads: List[np.ndarray]):
        policy_weight, v_weight = grads[0], grads[1]
        nn.utils.convert_parameters.vector_to_parameters(policy_weight, self.policy.parameters())
        nn.utils.convert_parameters.vector_to_parameters(v_weight, self.v.parameters())

class TRPO:
    def __init__(self, 
        delta: float, gamma: float, lamda: float, rtol: float, atol: float,
        damping_factor: float, max_cg: int, alpha: float, max_search: int,
        train_v_iters: int, enable_cuda: bool,
        **kwargs
    ):
        self.delta = delta
        self.gamma = gamma
        self.lamda = lamda
        self.rtol = rtol
        self.atol = atol
        self.damping_factor = damping_factor
        self.max_cg = max_cg
        self.alpha = alpha
        self.max_search = max_search
        self.train_v_iters = train_v_iters
        self.enable_cuda = enable_cuda
        self.networks = ApproxContainer(**kwargs)
        action_type = kwargs['action_type']
        if action_type == 'continu':
            self.action_distirbution_cls = GaussDistribution
        elif action_type == 'discret':
            self.action_distirbution_cls = CategoricalDistribution
        else:
            raise ValueError(f'Unknown action_type: {action_type}')

    def compute_gradient(self, data: Dict[str, torch.Tensor], iteration: int):
        start_time = time.time()
        obs, act, rew, obs2, done, time_limited = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['time_limited']
        done = torch.logical_or(done, time_limited)  # Current workaround: avoid self-bootstrap
        val: torch.Tensor = self.networks.v(obs)
        # Run _estimate_advantage on gpu may not be benificial
        # Or make _estimate_advantage work better with cuda
        adv, ret = self._estimate_advantage(obs, obs2, rew, done, val.detach())
        if self.enable_cuda:
            self.networks.cuda()
            obs, act, adv, ret = obs.cuda(), act.cuda(), adv.cuda(), ret.cuda()

        # pi
        with torch.no_grad():
            logits_old = self.networks.policy(obs)
            # print('std:', logits_old[...,1].mean().item())
        pi_old = self._get_distribution(logits=logits_old)
        logp_old = pi_old.log_prob(act)

        def get_surrogate_advantage(logp: torch.Tensor):
            return torch.mean(
                torch.exp(logp - logp_old) * adv
            )

        logits = self.networks.policy(obs)
        pi = self._get_distribution(logits=logits)
        surrogate_advantage = get_surrogate_advantage(pi.log_prob(act))
        g_params = torch.autograd.grad(surrogate_advantage, self.networks.policy.parameters(), retain_graph=True)
        # FIXME: CNN layer's g would be non-contiguous, needing further investigation
        # Current workaround is making sure it's contiguous
        g_params = [g_param.contiguous() for g_param in g_params]
        g_vec = nn.utils.convert_parameters.parameters_to_vector(g_params)
        x0_vec = torch.zeros_like(g_vec)
        d_kl = pi.kl_divergence(pi_old).mean()
        def hvp(f: torch.Tensor, x: torch.Tensor):
            # FIXME: CNN layer's g would be non-contiguous, needing further investigation
            # Current workaround is making sure it's contiguous
            g_params = torch.autograd.grad(f, self.networks.policy.parameters(), create_graph=True)
            g_params = [g_param.contiguous() for g_param in g_params]
            g_vec = nn.utils.convert_parameters.parameters_to_vector(g_params)
            hvp_params = torch.autograd.grad(
                torch.dot(g_vec, x),
                self.networks.policy.parameters(), retain_graph=True
            )
            # FIXME: CNN layer's hvp would be non-contiguous, e.g. with stride of (9, 144, 3, 1), needing further investigation
            # Current workaround is making sure it's contiguous
            hvp_params = [hvp_param.contiguous() for hvp_param in hvp_params]
            return nn.utils.convert_parameters.parameters_to_vector(hvp_params)

        def cg_func(x: torch.Tensor):
            return hvp(d_kl, x).add_(x, alpha=self.damping_factor)

        x_vec, _ = self._conjugate_gradient(cg_func, g_vec, x0_vec, self.rtol, self.atol, self.max_cg)

        weight_old = nn.utils.convert_parameters.parameters_to_vector(self.networks.policy.parameters())
        new_policy = self._create_new_policy()
        trpo_step = torch.sqrt(2*self.delta / (torch.dot(g_vec, x_vec) + EPSILON)) * x_vec
        def update_policy(alpha: float):
            weight_new = weight_old.add(trpo_step, alpha=alpha)
            nn.utils.convert_parameters.vector_to_parameters(weight_new, new_policy.parameters())

        for i in range(self.max_search):
            update_policy(self.alpha**i)
            # with torch.no_grad():
            logits_new = new_policy(obs)
            pi_new = self._get_distribution(logits=logits_new)
            logp_new = pi_new.log_prob(act)

            if get_surrogate_advantage(logp_new) > 0 and pi_new.kl_divergence(pi_old).mean() < self.delta:
                break
        else:
            print("fail to improve policy!")
            new_policy = self.networks.policy

        # v loss
        for i in range(self.train_v_iters):
            val = self.networks.v(obs)
            self.networks.v_optimizer.zero_grad()
            v_loss = F.mse_loss(val, ret)
            v_loss.backward()
            self.networks.v_optimizer.step()
        v_loss = v_loss.item()
        val_avg = val.detach().mean().item()

        if self.enable_cuda:
            new_policy.cpu()
            self.networks.cpu()

        with torch.no_grad():
            weight_list = (
                nn.utils.convert_parameters.parameters_to_vector(new_policy.parameters()),
                nn.utils.convert_parameters.parameters_to_vector(self.networks.v.parameters()),
            )
        end_time = time.time()

        tb_info = {}
        tb_info[tb_tags["loss_critic"]] = v_loss
        tb_info[tb_tags["critic_avg_value"]] = val_avg
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        tb_info[tb_tags["loss_actor"]] = -surrogate_advantage.item()
        return weight_list, tb_info

    def _create_new_policy(self):
        new_policy = deepcopy(self.networks.policy)
        for p in new_policy.parameters():
            p.requires_grad_(False)
        return new_policy

    @staticmethod
    def _conjugate_gradient(
        Ax: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, x: torch.Tensor,
        rtol: float, atol: float, max_cg: int
    ):
        """Conjugate gradient method

        Solve $Ax=b$ where $A$ is a positive definite matrix.
        Refer to https://en.wikipedia.org/wiki/Conjugate_gradient_method.

        Args:
            Ax (Callable[[torch.Tensor], torch.Tensor]): Function to calculate $Ax$, return value shape (S,)
            b (torch.Tensor): b, shape (S,)
            x (torch.Tensor): Initial x value, shape (S,)
            rtol (float): Relative tolerance
            atol (float): Absolute tolerance
            max_cg (int): Maximum conjugate gradient iterations

        Raises:
            ValueError: When failed to converge within max_cg iterations

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Solution of $Ax=b$, residue
        """
        zero = x.new_zeros(())
        r = b - Ax(x)
        if torch.allclose(r, zero, rtol=rtol, atol=atol): 
            print(f'mode0: {r.norm(2)} ?')
            return x, r

        r_dot = torch.dot(r, r)
        p = r.clone()
        for i in range(max_cg):
            Ap = Ax(p)
            alpha = r_dot / (torch.dot(p, Ap) + EPSILON)
            x = x.add_(p, alpha=alpha)
            r = r.add_(Ap, alpha=-alpha)
            if torch.allclose(r, zero, rtol=rtol, atol=atol): 
                print(f'mode1: {i} converged {r.norm(2)}')
                return x, r
            r_dot, r_dot_old = torch.dot(r, r), r_dot
            beta = r_dot / r_dot_old
            p = r.add(p, alpha=beta)
        print(f'mode2: max_cg {r.norm(2)}, {b.norm(2)}, {r.norm(2) / b.norm(2)}')
        return x, r
        # raise ValueError("_conjugate_gradient failed to converge within max_cg iterations")

    def _get_distribution(self, logits: torch.Tensor):
        return self.action_distirbution_cls(logits)

    def _estimate_advantage(self, obs: torch.Tensor, obs2: torch.Tensor, rew: torch.Tensor, done: torch.Tensor, val: torch.Tensor):
        gamma, lamda = self.gamma, self.lamda

        def discounted_cumsum(vec: torch.Tensor, gamma: float):
            cumsum: np.ndarray = scipy.signal.lfilter([1], [1, -gamma], vec.numpy()[::-1])[::-1]
            return torch.from_numpy(cumsum.copy()).float()

        def cal_traj(rew: torch.Tensor, value: torch.Tensor):
            delta = rew + gamma  * value[1:] - value[:-1]
            advantage = discounted_cumsum(delta, lamda*gamma)
            ret = discounted_cumsum(rew, gamma)
            return advantage, ret
        
        batch_size = obs.size(0)
        done_index = done.nonzero(as_tuple=False)[:, 0].tolist()
        if done[batch_size-1] == 0:
            done_index.append(batch_size-1)
        traj_num = len(done_index)

        adv_buf = torch.empty(batch_size)
        ret_buf = torch.empty(batch_size)

        start = 0
        for i in range(traj_num):
            end = done_index[i]+1

            traj_len = end - start
            traj_value = torch.empty(traj_len+1)
            traj_value[:-1] = val[start:end]
            if done[end-1] == 1:
                traj_value[-1] = 0
            else:
                with torch.no_grad():
                    traj_value[-1] = self.networks.v(obs2[end-1:end]) 
            adv_buf[start:end], ret_buf[start:end]  = cal_traj(rew[start:end], traj_value)
            start = end
        
        adv_buf_std, adv_buf_mean = torch.std_mean(adv_buf)
        adv_buf = adv_buf.sub_(adv_buf_mean).div_(adv_buf_std)
        return adv_buf, ret_buf

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.networks.load_state_dict(state_dict)
