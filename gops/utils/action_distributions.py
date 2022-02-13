"""


"""
import torch


class GaussDistribution():
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1
        )
        self.act_high_lim = torch.tensor([1.])
        self.act_low_lim = torch.tensor([-1.])

    def sample(self):
        action = self.gauss_distribution.sample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action)/1.0000001 + (
                    self.act_high_lim + self.act_low_lim) / 2
        return action_limited

    def rsample(self):
        action = self.gauss_distribution.rsample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action)/1.0000001 + (
                    self.act_high_lim + self.act_low_lim) / 2
        return action_limited

    def log_prob(self, action_limited) -> torch.Tensor:
        action = torch.atanh(
            (2 * action_limited - (self.act_high_lim + self.act_low_lim)) / (self.act_high_lim - self.act_low_lim))
        log_prob = self.gauss_distribution.log_prob(action) - torch.log((self.act_high_lim - self.act_low_lim) * (1 - torch.pow(torch.tanh(action), 2))).sum(-1)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) + (self.act_high_lim + self.act_low_lim) / 2

    def kl_divergence(self, other:'GaussDistribution') -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(self.gauss_distribution, other.gauss_distribution)


class CategoricalDistribution:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.cat = torch.distributions.Categorical(logits=logits)

    def sample(self):
        return self.cat.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() > 1: action = action.squeeze(1)
        return self.cat.log_prob(action)

    def entropy(self):
        return self.cat.entropy()

    def mode(self):
        return torch.argmax(self.logits, dim=-1)

    def kl_divergence(self, other: 'CategoricalDistribution'):
        return torch.distributions.kl.kl_divergence(self.cat, other.cat)


class DiracDistribution():
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        return self.logits

    def mode(self):
        return self.logits


class ValueDiracDistribution():
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        return torch.argmax(self.logits, dim=-1)

    def mode(self):
        return torch.argmax(self.logits, dim=-1)


if __name__ == "__main__":
    logits = torch.tensor([0.0, 1.0])
    act_dist = GaussDistribution(logits)
    for i in range(10):
        act = act_dist.rsample()
        print('act', act)
        print('log_prob', act_dist.log_prob(act))
