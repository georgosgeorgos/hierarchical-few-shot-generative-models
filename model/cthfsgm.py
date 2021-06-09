import os
import sys

import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.chfsgm import CHFSGM
from model.layer.tlayers_conv import (AttentivePosteriorC, AttentivePriorC, AttentiveStatistiC)


class CTHFSGM(CHFSGM):
    """
    Hierarchical Few-Shot Generative Model.
    We implement a hierarchy over c.
    The joint prior can be written as:
    p(Z, c) = \prod_l p(Z_l | Z_{l+1}, c_l) p(c_l | c_{l+1}, Z_{l+1}).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        statistic_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.n_features,
            self.activation,
            self.dropout_sample,
            self.aggregation_mode
        )
        # q(c | X)
        self.statistic = AttentiveStatistiC(*statistic_args)

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

    def conditional_sample_cq_hierarchy(self, x):
        """
        Sample the model given a context X.
        Sample are obtained in one-shot
        using the approximate posterior over c
        and the generative model over z.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        bs = x.shape[0]
        ns = x.shape[1]

        x = x.view(bs*ns, -1, self.img_dim, self.img_dim)
        h = self.encoder(x)
        h = self.proj(h)
        # use mean
        cq, _, _, att = self.statistic(h, bs, ns)
        att_lst = [att]
        
        cq_samples = [cq]
        
        zp_mean, zp_logvar = self.prior_z[-1](None, cq, bs, ns)
        zpd = self.normal(zp_mean, zp_logvar)
        zp = zpd.sample()
        zp_samples = [zp]

        for td in reversed(range(self.n_stochastic - 1)):

            cq, _, _, att, _ = self.posterior_c[td](h, zp, cq, bs, ns)            
            cq_samples.append(cq)
            att_lst.append(att)

            zp_mean, zp_logvar = self.prior_z[td](zp, cq, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            zp = zpd.sample()
            zp_samples.append(zp)

        xp_lst = []
        cq_tmp = []
        zp_tmp = []
        ctmp = torch.zeros(cq.size()).cuda()
        ztmp = torch.zeros(zp.size()).cuda()
        
        cq_tmp = [ctmp for _ in range(self.n_stochastic)]
        zp_tmp = [ztmp for _ in range(self.n_stochastic)]
        for i in range(self.n_stochastic):
            
            cq_tmp[i]=cq_samples[i]
            cs = torch.cat(cq_tmp, dim=1)
            
            zp_tmp[i]=zp_samples[i]
            zs = torch.cat(zp_tmp, dim=1)

            xp = self.observation_decoder(zs, cs, bs, ns)
            xp = self.likelihood.sample(xp)

            xp_lst.append(xp)
        xp = xp.view(bs, ns, -1, self.img_dim, self.img_dim)
        return {"xp": xp, "att": att_lst, "xp_lst": xp_lst}
        