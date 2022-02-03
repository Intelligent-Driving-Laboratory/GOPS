#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gops.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution, CategoricalDistribution
import numpy as np  # Matrix computation library
import torch
import torch.nn as nn


class Action_Distribution():
    def __init__(self):
        super().__init__()

    def get_act_dist(self, logits):
        act_dist_cls = getattr(self, 'action_distirbution_cls')
        has_act_lim = hasattr(self, 'act_high_lim')

        act_dist = act_dist_cls(logits)
        if has_act_lim:
            act_dist.act_high_lim = getattr(self, 'act_high_lim')
            act_dist.act_low_lim = getattr(self, 'act_low_lim')

        return act_dist
