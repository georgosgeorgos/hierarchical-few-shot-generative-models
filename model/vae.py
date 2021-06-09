import os
import sys

import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.base import BaseModel
from model.layer.layers_fc import (ObservationDecoder, PosteriorZ, PreProj,
                                   PriorZ, SharedConvEncoder, StatistiC)


class VAE(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # convolutional encoder
        encoder_args = (self.ch_enc, self.activation)
        self.encoder = SharedConvEncoder(*encoder_args)

        self.proj = PreProj(self.hidden_dim, self.n_features, self.activation)
        self.drop = nn.Dropout(0.1)

        z_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.z_dim,
            self.h_dim,
            self.activation,
        )
        # inference networks
        self.posterior_z = nn.ModuleList(
            [PosteriorZ(*z_args) for _ in range(self.n_stochastic)]
        )

        # latent decoders
        self.prior_z = nn.ModuleList(
            [PriorZ(*z_args) for _ in range(self.n_stochastic-1)]
        )

        # observation decoder
        obs_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.z_dim,
            self.h_dim,
            self.n_stochastic,
            1,
            self.activation,
        )
        self.observation_decoder = ObservationDecoder(*obs_args)

    def top_down_stochastic_inference(self, 
                                      hq: torch.Tensor, 
                                      out: dict,
                                      bs=None,
                                      ns=None
                                      ) -> dict:
        """
        Top-down stochastic inference of for z.
        
        Args:
            hq:  encoded data X.
            out: collection of lists with priors for z, 
                 approximate posteriors for z and c, 
                 representations for X and x.
        Returns: 
            Collection with updated priors for z, 
            updated approximate posteriors for z. 
        """
        cq = None
        zq = None
        for td in range(self.n_stochastic): # L, L-1,...,1
            # q(z_{l} | z_{l+1}, c, h)
            zq_mean, zq_logvar = self.posterior_z[td](hq, zq, cq, bs, ns)
            zqd = self.normal(zq_mean, zq_logvar)
            zq = zqd.rsample()

            out["zqd"].append(zqd)
            out["zqs"].append(zq)
            
        zq = None
        for td in range(self.n_stochastic):
            # p(z_{l} | z_{l+1}, c)
            if td == 0:
                zpd = self.standard_normal(out["zqs"][0])
            else:
                zp_mean, zp_logvar = self.prior_z[td-1](zq, cq, bs, ns)
                zpd = self.normal(zp_mean, zp_logvar)
            
            zq = out["zqs"][td]

            out["zpd"].append(zpd)
        return out

    def forward(self, x: torch.Tensor):
        """
        zqd: approximate posterior (q) distribution (d) over samples (z).
        zq: sample from approximate posterior (q) over samples (z).
        zpd: prior/generative (p) distribution over samples (z).

        Args:
            x: batch of (small) datasets.
        Returns: collection of lists with priors for z, 
                 approximate posteriors for z and c, 
                 representations for X and x.
        """
        bs=x.shape[0] 
        ns=x.shape[1]

        lst = ["zqs", # inference samples 
               "zqd", # inference networks for z
               "zpd", # prior z
               ]                 
        out={ l: [] for l in lst }

        # convolutional encoder
        h = self.encoder(x)
        h = self.proj(h)
        # top-down stochastic inference - same direction generation
        out = self.top_down_stochastic_inference(h, out, bs, ns)    
        # observation decoder
        zs = torch.cat(out["zqs"], dim=1)
        xp = self.observation_decoder(zs, None, bs, ns)

        out["x"] = x.view(bs, ns, 1, 28, 28)
        out["xp"] = xp.view(bs, ns, 1, 28, 28)
        return out

    def loss(self, 
             out: dict, 
             weight: float=1.0,
             free_bits: bool = False, bs=None, ns=None
             ) -> (torch.Tensor, torch.Tensor):
        """
        Perform variational inference and compute the loss.

        Args:
            out: collection of lists with priors for z, 
                 approximate posteriors for z and c, 
                 representations for X and x.
            weight: reweights the lower-bound.
        Returns:
            Loss and variational lowerbound.
        """
        bs=out["x"].shape[0] 
        ns=out["x"].shape[1]

        den = bs * ns
        
        x = out["x"].view(-1, 1, 28, 28)
        xp = out["xp"].view(-1, 1, 28, 28)
        # Likelihood for observation model
        px = td.Bernoulli(probs=xp)
        logpx = px.log_prob(x)
        logpx = logpx.sum() / den

        # kl terms over c and z
        kl_z = 0

        zqd = out["zqd"]
        # prior over z for all layers
        zpd = out["zpd"]

        for l in range(self.n_stochastic):
            kl_z += td.kl_divergence(zqd[l], zpd[l]).sum()
            
        kl_z /= den
        kl = kl_z
        
        # Variational lower bound and weighted loss
        vlb = logpx - kl
        w_vlb = ( (weight * logpx) - (kl / weight) )
        loss = - w_vlb
        
        return {"loss": loss, "vlb": vlb, "logpx": logpx, "kl_c": torch.tensor([0]), "kl_z": kl_z}

    def unconditional_sample(self, bs=10, ns=5, device='cuda'):
        """
        Sample the model unconditionally.

        Args:
            ns: number of samples to generate.
        Returns:
            Sample from the model.
        """
        zp_samples = []

        zp = torch.randn(bs, ns, self.z_dim).to(device)
        zp_samples.append(zp.view(-1, self.z_dim))
        for td in range(self.n_stochastic-1):
            # p(z_l | z_{l+1})
            # generation time z_{l+1} is sampled from the prior
            zp_mean, zp_logvar = self.prior_z[td](zp, None, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            zp = zpd.sample()
            zp_samples.append(zp)

        # observation decoder p(x | z_{1:L}, c)
        zs = torch.cat(zp_samples, dim=1)
        xp = self.observation_decoder(zs, None, bs, ns)
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def compute_mll(self,
                    out: dict,
                    ) -> dict:
        """
        Perform variational inference and compute the loss.
        Args:
            out: collection of lists with priors for z and c, 
                 approximate posteriors for z and c, 
                 representations for X and x.
        Returns:
        """
        bs = out["x"].shape[0]
        ns = out["x"].shape[1]
        
        x = out["x"].view(-1, 1, 28, 28)
        xp = out["xp"].view(-1, 1, 28, 28)
        # Likelihood for observation model
        px = td.Bernoulli(xp)
        logpx = px.log_prob(x).sum(-1).sum(-1)
        
        # KL Divergence terms over c and z
        kl_z = 0

        # posterior over z for all layers
        zqd = out["zqd"]
        zqs = out["zqs"]
        # prior over z for all layers
        zpd = out["zpd"]

        for l in range(self.n_stochastic):
            #kl_z += td.kl_divergence(zqd[l], zpd[l]).sum(-1)
            kl_z += (zqd[l].log_prob(zqs[l]) - zpd[l].log_prob(zqs[l])).sum(-1)
        
        logpx = logpx.view(bs, -1)#.sum(-1) / ns
        kl_z = kl_z.view(bs, -1)#.sum(-1) / ns
        
        kl = kl_z
        # Variational lower bound and weighted loss
        vlb = logpx - kl
        return {"vlb": vlb.squeeze()}
