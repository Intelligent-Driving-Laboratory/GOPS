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

    def sample(self):
        return self.gauss_distribution.sample()

    def rsample(self):
        return self.gauss_distribution.rsample()

    def log_prob(self, action) -> torch.Tensor:
        return self.gauss_distribution.log_prob(action)

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return self.mean

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
