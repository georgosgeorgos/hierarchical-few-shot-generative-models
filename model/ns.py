import os

import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.base import BaseModel
from model.layer.layers_fc import (PosteriorZ, PriorZ, ObservationDecoder,
                      SharedConvEncoder, StatistiC, PreProj)

import torch.distributions as td


# Model
class NS(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # convolutional encoder
        encoder_args = (self.ch_enc, self.activation)
        self.encoder = SharedConvEncoder(*encoder_args)

        # statistic network
        statistic_args = (
            self.hidden_dim,
            self.c_dim,
            self.n_features,
            self.activation,
            False,
            self.aggregation_mode
        )
        self.statistic = StatistiC(*statistic_args)

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
            [PriorZ(*z_args) for _ in range(self.n_stochastic)]
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
        c_mean, c_logvar, _, _ = self.statistic(h, bs, ns)
        cqd = self.normal(c_mean, c_logvar)
        cq = cqd.rsample()

        # input cq for dimensions
        cpd = self.standard_normal(cq)
        
        out["cqd"] = [cqd]
        out["cqs"] = [cq]
        out["cpd"] = [cpd]
        return cq, out

    def top_down_stochastic_inference(self, 
                                      hq: torch.Tensor, 
                                      out: dict,
                                      bs=None,
                                      ns=None
                                      ) -> dict:
        """
        Top-down stochastic inference of for z.
        
        Args:
            cq:  sample from q(c | X).
            hq:  encoded data X.
            out: collection of lists with priors for z, 
                 approximate posteriors for z and c, 
                 representations for X and x.
        Returns: 
            Collection with updated priors for z, 
            updated approximate posteriors for z. 
        """
        cq = out['cqs'][0]
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
            zp_mean, zp_logvar = self.prior_z[td](zq, cq, bs, ns)
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
        # statistic network q(c_L | X)
        cq, out = self.statistics_inference(h, out, bs, ns)
        # top-down stochastic inference - same direction generation
        out = self.top_down_stochastic_inference(h, out, bs, ns)
        # observation decoder
        zs = torch.cat(out["zqs"], dim=1)
        xp = self.observation_decoder(zs, cq, bs, ns)
        
        out["x"] = x.view(bs, ns, 1, 28, 28)
        out["xp"] = xp.view(bs, ns, 1, 28, 28)
        return out

    def reconstruction(self, x):
        bs=x.shape[0] 
        ns=x.shape[1]
        xp = self.forward(x)["xp"]
        xp = xp.view(bs*ns, -1, self.img_dim, self.img_dim)
        rec = self.likelihood.sample(xp)
        rec = rec.view(bs, ns, -1, self.img_dim, self.img_dim)
        return rec

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
        # likelihood for observation model
        px = td.Bernoulli(probs=xp)
        logpx = px.log_prob(x)
        logpx = logpx.sum() / den

        # kl terms over c and z
        kl_c = 0
        kl_z = 0

        # posterior over c 
        cqd = out["cqd"]
        # # prior over c 
        cpd = out["cpd"]
        # posterior over z for all layers
        zqd = out["zqd"]
        # prior over z for all layers
        zpd = out["zpd"]

        kl_c = td.kl_divergence(cqd[0], cpd[0]).sum()
        for l in range(self.n_stochastic):
            kl_z += td.kl_divergence(zqd[l], zpd[l]).sum()
            
        # is it correct for kl_c?
        kl_c /= den
        kl_z /= den
        kl = kl_c + kl_z
        
        # Variational lower bound and weighted loss
        vlb = logpx - kl
        w_vlb = ( (weight * logpx) - (kl / weight) )
        loss = - w_vlb
        
        return {"loss": loss, "vlb": vlb, "logpx": logpx, "kl_c": kl_c, "kl_z": kl_z}

    def unconditional_sample(self, bs=10, ns=5, device='cuda'):
        """
        Sample the model.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        # sample the autoregressive prior
        zp_samples = []
        zp = None 
        cq = torch.randn(bs, 1, self.c_dim).to(device)
        #torch.randn(bs, ns, self.z_dim).to(h.device)
        #zp_samples.append(zp)
        for td in range(self.n_stochastic):
            # p(z_l | z_{l+1})
            # generation time z_{l+1} is sampled from the prior
            zp_mean, zp_logvar = self.prior_z[td](zp, cq, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            zp = zpd.sample()
            zp_samples.append(zp)

        # observation decoder p(x | z_{1:L}, c)
        zs = torch.cat(zp_samples, dim=1)
        xp = self.observation_decoder(zs, cq, bs, ns)
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def conditional_sample_cqL(self, x):
        """
        Sample the model given a context X.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        bs=x.shape[0] 
        ns=x.shape[1]

        x = x.view(-1, 1, 28, 28)
        h = self.encoder(x)
        h = self.proj(h)
        # use mean
        cq, _, _, _ = self.statistic(h, bs, ns)

        # latent decoders
        # sample the autoregressive prior
        zp_samples = []
        zp = None 
        #torch.randn(bs, ns, self.z_dim).to(h.device)
        #zp_samples.append(zp)
        for td in range(self.n_stochastic):
            # p(z_l | z_{l+1})
            # generation time z_{l+1} is sampled from the prior
            zp_mean, zp_logvar = self.prior_z[td](zp, cq, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            zp = zpd.sample()
            zp_samples.append(zp)

        # observation decoder p(x | z_{1:L}, c)
        zs = torch.cat(zp_samples, dim=1)
        xp = self.observation_decoder(zs, cq, bs, ns)
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def conditional_sample_mcmc_v1(self, x, mc=10):
        """
        Sample the model given a context X.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        lst = ["zqs", # inference samples 
               "zqd", # inference networks for z
               "zpd", # prior z
               ]                  
        bs=x.shape[0] 
        ns=x.shape[1]

        x = x.view(-1, 1, 28, 28)
        xp = self.conditional_sample_cqL(x)["xp"]    

        # convolutional encoder
        hc = self.encoder(x)
        hc = self.proj(hc)
        cq, _, _, _ = self.statistic(hc, bs, ns)

        for _ in range(mc):

            out={ l: [] for l in lst }
            out["cqs"] = [cq]

            hz = self.encoder(xp)
            hz = self.proj(hz)

            out = self.top_down_stochastic_inference(hz, out, bs, ns)    
            # observation decoder
            zs = torch.cat(out["zqs"], dim=1)
            xp = self.observation_decoder(zs, cq, bs, ns)
            
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def conditional_sample_mcmc_v2(self, x, mc=10, mode="use_q"):
        """
        Sample the model given a context X.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        lst = ["zqs", # inference samples 
               "zqd", # inference networks for z
               "zpd", # prior z
               ]                  
        bs=x.shape[0] 
        ns=x.shape[1]

        x = x.view(-1, 1, 28, 28)
        xp = self.conditional_sample_cqL(x)["xp"]    

        # convolutional encoder
        hc = self.encoder(x)
        hc = self.proj(hc)
         
        for _ in range(mc):

            out={ l: [] for l in lst }
            hz = self.encoder(xp)
            hz = self.proj(hz)

            h = torch.cat([hc, hz], 1)
            # statistic network q(c_L | X)
            cq_mean, cq_logvar, _, _ = self.statistic(h, bs, ns)
            cqd = self.normal(cq_mean, cq_logvar)
            cq = cqd.sample() 

            # top-down stochastic inference - same direction generation
            if mode == "use_q":
                out['cqs'] = [cq]
                out = self.top_down_stochastic_inference(hz, out, bs, ns)
                zs = torch.cat(out["zqs"], dim=1)

            elif mode == "use_p":
                zp_samples = []
                zp = None
                for td in range(self.n_stochastic):
                    zp_mean, zp_logvar = self.prior_z[td](zp, cq, bs, ns)
                    zpd = self.normal(zp_mean, zp_logvar)
                    zp = zpd.sample()
                    zp_samples.append(zp)
                zs = torch.cat(zp_samples, dim=1)

            # observation decoder
            xp = self.observation_decoder(zs, cq, bs, ns)
            
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

        # posterior over c for all layers
        cqd = out["cqd"]
        cqs = out["cqs"]
        # prior over c for all layers
        cpd = out["cpd"]
        # posterior over z for all layers
        zqd = out["zqd"]
        zqs = out["zqs"]
        # prior over z for all layers
        zpd = out["zpd"]

        kl_c = (cqd[0].log_prob(cqs[0]) - cpd[0].log_prob(cqs[0])).sum(-1) 
        #td.kl_divergence(cqd[0], cpd[0]).sum(-1)
        for l in range(self.n_stochastic):
            #kl_z += td.kl_divergence(zqd[l], zpd[l]).sum(-1)
            kl_z += (zqd[l].log_prob(zqs[l]) - zpd[l].log_prob(zqs[l])).sum(-1)
        
        logpx = logpx.view(bs, -1).sum(-1)
        kl_z = kl_z.view(bs, -1).sum(-1)
        kl_c = kl_c.squeeze()
        
        kl = kl_c + kl_z
        # Variational lower bound and weighted loss
        vlb = logpx - kl
        return {"vlb": vlb.squeeze()}

