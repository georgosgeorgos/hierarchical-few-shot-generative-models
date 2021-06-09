import os

import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.ns import NS
from model.layer.tlayers_fc import AttentiveStatistiC

import torch.distributions as td

# Model
class TNS(NS):
    """
    Neural Statistician with learnable aggregation over set.
    The learnable aggregation is implemented throught a variant of self-attention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # statistic network
        statistic_args = (
            self.hidden_dim,
            self.c_dim,
            self.n_features,
            self.activation,
            False,
            self.aggregation_mode
        )
        self.statistic = AttentiveStatistiC(*statistic_args)

    def statistics_inference(self, 
                             h: torch.Tensor, 
                             out: dict, 
                             bs=None, 
                             ns=None
                             ) -> (torch.Tensor, dict):
        """
        Compute initial statistics using q(c | X).

        Args: 
            h: initial encoding for the data X.
            out: collection of lists with priors for z, 
                 approximate posteriors for z and c, 
                 representations for X and x.
        Returns:
            cq: sample from approximate posterior q(c | X).
            rcq: representations over the samples in X.
            out: collection with updated approximate posterior for c.
        """
        c_mean, c_logvar, _, att = self.statistic(h, bs, ns)
        cqd = self.normal(c_mean, c_logvar)
        cq = cqd.rsample()

        # input cq for dimensions
        cpd = self.standard_normal(cq)
        
        out["cqd"] = [cqd]
        out["cqs"] = [cq]
        out["cpd"] = [cpd]
        return cq, out
        