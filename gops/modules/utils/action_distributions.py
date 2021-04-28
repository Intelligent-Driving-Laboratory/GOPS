"""


"""
import torch

class GaussDistribution():
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.split(logits, dim=1)
        self.gauss_distribution = torch.distributions.Normal(self.mean, self.std)

    def sample(self):
        return self.gauss_distribution.sample()

    def log_prob(self, action):
        return self.gauss_distribution.log_prob(action)

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return self.mean


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
