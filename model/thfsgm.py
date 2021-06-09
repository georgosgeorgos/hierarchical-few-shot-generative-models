import os
import sys

import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.hfsgm import HFSGM
from model.layer.tlayers_fc import (AttentivePosteriorC, AttentivePosteriorZ,
                           AttentivePriorC, AttentivePriorZ,
                           AttentiveStatistiC)


class THFSGM(HFSGM):
    """
    Hierarchical Few-Shot Generative Model.
    We implement a hierarchy over c.
    The joint prior can be written as:
    p(Z, c) = \prod_l p(Z_l | Z_{l+1}, c_l) p(c_l | c_{l+1}, Z_{l+1}).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
        statistic_args = (
            self.hidden_dim,
            self.c_dim,
            self.n_features,
            self.activation,
            self.dropout_sample,
            self.aggregation_mode
        )
        # q(c | X)
        self.statistic = AttentiveStatistiC(*statistic_args)

        z_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.z_dim,
            self.h_dim,
            self.activation,
        )
        # q(z_l | z_{l+1}, c_l, h_l) ladder formulation
        posterior_z = []
        # p(z_l | z_{l+1}, c_l)
        prior_z = []
        # initialize hierarchy for z
        for _ in range(self.n_stochastic):
            posterior_z.append(AttentivePosteriorZ(*z_args))
            prior_z.append(AttentivePriorZ(*z_args))

        self.posterior_z = nn.ModuleList(posterior_z)
        self.prior_z = nn.ModuleList(prior_z)


        c_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.z_dim,
            self.h_dim,
            self.activation,
            self.aggregation_mode,
            self.ladder
        )
        # q(c_l | z_{l+1}, c_{l+1}, h_l)
        posterior_c = []
        prior_c = []
        # initialize hierarchy for c
        for _ in range(self.n_stochastic - 1):
            posterior_c.append(AttentivePosteriorC(*c_args))
            prior_c.append(AttentivePriorC(*c_args))
        
        self.posterior_c = nn.ModuleList(posterior_c)
        self.prior_c = nn.ModuleList(prior_c)
        