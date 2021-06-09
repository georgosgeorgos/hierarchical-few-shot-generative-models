import os
import sys

import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from model.base import BaseModel
from model.layer.layers_fc import (PosteriorC, PriorC, PosteriorZ, PriorZ, ObservationDecoder,
                      SharedConvEncoder, StatistiC, PreProj, EncoderBottomUp)


class HFSGM(BaseModel):
    """
    Hierarchical Few-Shot Generative Model.
    We implement a hierarchy over c.
    The joint prior can be written as:
    p(Z, c) = \prod_l p(Z_l | Z_{l+1}, c_l) p(c_l | c_{l+1}).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # convolutional encoder
        encoder_args = (self.ch_enc, self.activation)
        self.encoder = SharedConvEncoder(*encoder_args)

        self.proj = PreProj(self.hidden_dim, self.n_features, self.activation)
        self.drop = nn.Dropout(0.1)

        # h_args = (self.num_layers, self.hidden_dim, self.activation)
        # self.encoder_bu = nn.ModuleList([EncoderBottomUp(*h_args) for _ in range(self.n_stochastic)])
        
        statistic_args = (
            self.hidden_dim,
            self.c_dim,
            self.n_features,
            self.activation,
            self.dropout_sample,
            self.aggregation_mode
        )
        # q(c | X)
        self.statistic = StatistiC(*statistic_args)

        z_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.z_dim,
            self.h_dim,
            self.activation,
        )
        # q(z_l | z_{l+1}, c_l, h_l) ladder formulation
        # sharing of parameters between generative and inference path
        posterior_z = []
        # p(z_l | z_{l+1}, c_l)
        prior_z = []

        # initialize hierarchy for z
        for _ in range(self.n_stochastic):
            posterior_z.append(PosteriorZ(*z_args))
            prior_z.append(PriorZ(*z_args))

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
        # p(c_l | z_{l+1}, c_{l+1})
        prior_c = []

        # initialize hierarchy for c
        for _ in range(self.n_stochastic - 1):
            posterior_c.append(PosteriorC(*c_args))
            prior_c.append(PriorC(*c_args))

        self.posterior_c = nn.ModuleList(posterior_c)
        self.prior_c = nn.ModuleList(prior_c)

        # p(x | \bar z, \bar c)
        obs_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.z_dim,
            self.h_dim,
            self.n_stochastic,
            self.n_stochastic,
            self.activation,
        )
        self.observation_decoder = ObservationDecoder(*obs_args)

        self.lst = ["zqs", "zqd",  # posterior z
                    "cqs", "cqd",  # posterior c
                    "zpd",         # prior z
                    "cpd",         # prior c
                    "h"            # data
                    "att",         # attention over context
                    ]

    def statistics_inference(self, out, bs, ns):
        """
        Compute initial statistics using q(c_L | X).

        Args:
            h: initial encoding for the data X.
            out: collection of lists with priors for z and c,
                 approximate posteriors for z and c,
                 representations for X and x.
        Returns:
            cq: sample from approximate posterior q(c_L | X).
            rcq: representations over the samples in X.
            out: collection with updated prior and approximate posterior for c on layer L.
        """

        h = out["h"][-1]
        c_mean, c_logvar, _, _ = self.statistic(h, bs, ns)
        cqd = self.normal(c_mean, c_logvar)
        cq = cqd.rsample()

        # input cq for dimensions
        cpd = self.standard_normal(cq)

        out["cqd"].append(cqd)
        out["cqs"].append(cq)
        out["cpd"].append(cpd)
        return out

    def bottom_up_deterministic_inference(self,
                                          h: torch.Tensor,
                                          out: dict,
                                          bs=None,
                                          ns=None
                                          ) -> torch.Tensor:
        """
        Bottom-up deterministic inference.
        h_0 = x, h_l = g(h_{l-1}, c)

        Args:
            hq: output inference layer l-1.
            out: collection of lists with priors for z and c,
                 approximate posteriors for z and c,
                 representations for X and x.
        Returns:
            Collection with updated representations for X and x.
        """
        for bu in range(self.n_stochastic):  # 1,..., L-1, L
            h = self.encoder_bu[bu](h, bs, ns)
            out["h"].append(h)
        return out

    def top_down_stochastic_inference(self,
                                      out,
                                      bs,
                                      ns
                                      ):
        """
        Top-down stochastic inference of for z and c.
        z_l <- f(z_{l+1}, c_l, (h_l))
        c_l <- g(c_{l+1}, z_{l+1}, (h_l))

        Give q(cq_L | X) and representations rcq_L

        1) p(zp_L | cq_L) --> extract representations rzp_L
        2) q(zq_L | cq_L, h_L, g(rzp_L))

        repeat:
        3) p(cp_l | rcq_{l+1}, zq_{l+1})  --> extract representations rcp_l
        4) q(cq_l | zq_{l+1}, H_l, rcp_l) --> extract representations rcq_l

        5) p(zp_l | zq_{l+1}, cq_l) --> extract representations rzp_l
        6) q(zq_l | cq_l, h_l, g(rzp_l))

        Args:
            cq:  sample from q(c_L | X).
            rcq: representation over samples in X.
            out: collection of lists with priors for z and c,
                 approximate posteriors for z and c,
                 representations for X and x.
        Returns:
            Collection with updated priors for z and c,
            updated approximate posteriors for z and c.
        """

        h = out["h"][-1]
        cq = out["cqs"][0]
        
        if "hz" in out:
            hz = out["hz"][-1]
        else:
            hz = h

        zq = None
        zq_mean, zq_logvar = self.posterior_z[-1](hz, None, cq, bs, ns)
        zqd = self.normal(zq_mean, zq_logvar)
        zq = zqd.rsample()
        
        zp_mean, zp_logvar = self.prior_z[-1](None, cq, bs, ns)
        zpd = self.normal(zp_mean, zp_logvar)

        out["zqd"].append(zqd)
        out["zqs"].append(zq)
        out["zpd"].append(zpd)
        
        for td in reversed(range(self.n_stochastic - 1)):  # L-1,...,1

            # h = out["h"][td]
            # if "hz" in out:
            #     hz = out["hz"][td]
            # else:
            #     hz = h

            # inference over c
            cp_mean, cp_logvar, _, _, cp_feats = self.prior_c[td](zq, cq, bs, ns)
            cpd = self.normal(cp_mean, cp_logvar)
            
            # q( cq_l | cq_{l+1}, zq_{l+1} )
            if not self.ladder:
                cp_feats = cq
            cq_mean, cq_logvar, _, _, _ = self.posterior_c[td](h, zq, cp_feats, bs, ns)
            cqd = self.normal(cq_mean, cq_logvar)
            cq = cqd.rsample()
            
            out["cqd"].append(cqd)
            out["cqs"].append(cq)
            out["cpd"].append(cpd)

            # inference over z
            zp_mean, zp_logvar = self.prior_z[td](zq, cq, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            
            # q( z_l | zp_l, h_l, c_l )
            zq_mean, zq_logvar = self.posterior_z[td](hz, zq, cq, bs, ns)
            zqd = self.normal(zq_mean, zq_logvar)
            zq = zqd.rsample()

            # skip connections in z
            # for i in range(len(out["zqs"])):
            #     zq += out["zqs"][i]

            out["zqd"].append(zqd)
            out["zqs"].append(zq)
            out["zpd"].append(zpd)
        return out

    def forward(self, x: torch.Tensor) -> dict:
        """
        cqd: approximate posterior (q) distribution (d) over context (c).
        cq: sample from approximate posterior (q) over context (c).
        cpd: prior/generative (p) distribution (d) over context (c).

        zqd: approximate posterior (q) distribution (d) over samples (z).
        zq: sample from approximate posterior (q) over samples (z).
        zpd: prior/generative (p) distribution over samples (z).

        Args:
            x: batch of (small) datasets.
        Returns: collection of lists with priors for z and c,
                 approximate posteriors for z and c,
                 representations for X and x.
        """
        bs = x.shape[0]
        ns = x.shape[1]

        lst = ["zqs", "zqd",  # posterior z
               "cqs", "cqd",  # posterior c
               "zpd",         # prior z
               "cpd",         # prior c
               "att",         # attention over context
               "h",
               ]
        out = {l: [] for l in lst}

        # convolutional encoder
        h = self.encoder(x)
        h = self.proj(h)
        #h = self.drop(h)
        out["h"] = [h]
        #out = self.bottom_up_deterministic_inference(h, out, bs, ns)

        # statistic network q(c_L | X)
        out = self.statistics_inference(out, bs, ns)
        # top-down stochastic inference - same direction generation
        out = self.top_down_stochastic_inference(out, bs, ns)

        # observation decoder
        zs = torch.cat(out["zqs"], dim=1)
        cs = torch.cat(out["cqs"], dim=1)
        # cs = 0
        # for i in range(len(out["cqs"])):
        #     cs += out["cqs"][i]
        # cs /= len(out["cqs"])
        #cs = out["cqs"][-1] #torch.cat(out["cqs"], dim=-1)
        xp = self.observation_decoder(zs, cs, bs, ns)
        
        out["x"] = x.view(bs, ns, 1, 28, 28)
        out["xp"] = xp.view(bs, ns, 1, 28, 28)
        return out

    def loss(self,
             out: dict,
             weight: float = 1.0,
             free_bits: bool = False
             ) -> (torch.Tensor, torch.Tensor):
        """
        Perform variational inference and compute the loss.

        Args:
            out: collection of lists with priors for z and c,
                 approximate posteriors for z and c,
                 representations for X and x.
            weight: reweights the lower-bound.
            free_bits: min value for KL.
        Returns:
            Loss and variational lowerbound.
        """
        bs = out["x"].shape[0]
        ns = out["x"].shape[1]

        den = bs * ns

        x = out["x"].view(-1, 1, 28, 28)
        xp = out["xp"].view(-1, 1, 28, 28)
        # Likelihood for observation model
        px = td.Bernoulli(probs=xp)
        logpx = px.log_prob(x)
        logpx = logpx.sum() / den

        # KL Divergence terms over c and z
        kl_z = 0
        kl_c = 0

        # posterior over c for all layers
        cqd = out["cqd"]
        # # prior over c for all layers
        cpd = out["cpd"]
        # posterior over z for all layers
        zqd = out["zqd"]
        # prior over z for all layers
        zpd = out["zpd"]

        for l in range(self.n_stochastic):
            if l==0:
                # just for debugging
                kl_cL = td.kl_divergence(cqd[l], cpd[l]).sum()

            kl_c += td.kl_divergence(cqd[l], cpd[l]).sum()
            kl_z += td.kl_divergence(zqd[l], zpd[l]).sum()

        kl_c /= den        
        kl_z /= den
        kl = kl_c + kl_z

        # Variational lower bound and weighted loss
        vlb = logpx - kl
        w_vlb = (weight * logpx) - (kl / weight)
        loss = - w_vlb

        return {"loss": loss, "vlb": vlb, "logpx": logpx, "kl_z": kl_z, "kl_cL": kl_cL / den, "kl_c": kl_c}

    def unconditional_sample(self, bs=10, ns=5, device='cuda'):
        """
        Sample the model.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        # latent decoders
        zp_samples = []
        cp = torch.randn(bs, 1, self.c_dim).to(device)
        cp_samples = [cp.view(-1, self.c_dim)]

        zp_mean, zp_logvar = self.prior_z[-1](None, cp, bs, ns)
        zpd = self.normal(zp_mean, zp_logvar)
        zp = zpd.sample()
        zp_samples = [zp]
        for td in reversed(range(self.n_stochastic - 1)):
            
            cp, _, _, _= self.prior_c[td](zp, cp, bs, ns)            
            cp_samples.append(cp)
            
            zp_mean, zp_logvar = self.prior_z[td](zp, cp, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            zp = zpd.sample()
            zp_samples.append(zp)

        zs = torch.cat(zp_samples, dim=-1)
        cs = torch.cat(cp_samples, dim=-1)
        xp = self.observation_decoder(zs, cs, bs, ns)
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    
    def conditional_sample_cqL(self, x):
        """
        Sample the model given a context X.
        Sample are obtained in one-shot 
        conditioning only the top-layer q(c_L | X).

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        bs = x.shape[0]
        ns = x.shape[1]

        x = x.view(-1, 1, 28, 28)
        h = self.encoder(x)
        h = self.proj(h)
        # use mean
        cq, _, _, _ = self.statistic(h, bs, ns)

        # latent decoders
        cp_samples = [cq]
        
        zp_mean, zp_logvar = self.prior_z[-1](None, cq, bs, ns)
        zpd = self.normal(zp_mean, zp_logvar)
        zp = zpd.sample()
        zp_samples = [zp]
        #zp = torch.randn(bs, ns, self.z_dim).to(cq.device)
        cp = cq
        for td in reversed(range(self.n_stochastic - 1)):
            
            cp, _, _, _ = self.prior_c[td](zp, cp, bs, ns)            
            cp_samples.append(cp)
            
            zp_mean, zp_logvar = self.prior_z[td](zp, cp, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            zp = zpd.sample()
            zp_samples.append(zp)

        zs = torch.cat(zp_samples, dim=-1)
        cs = torch.cat(cp_samples, dim=-1)
        xp = self.observation_decoder(zs, cs, bs, ns)
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def conditional_sample_cq(self, x):
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

        x = x.view(-1, 1, 28, 28)
        h = self.encoder(x)
        h = self.proj(h)
        # use mean
        cq, _, _, _ = self.statistic(h, bs, ns)

        cq_samples = [cq]
        
        zp_mean, zp_logvar = self.prior_z[-1](None, cq, bs, ns)
        zpd = self.normal(zp_mean, zp_logvar)
        zp = zpd.sample()
        zp_samples = [zp]

        for td in reversed(range(self.n_stochastic - 1)):

            cq, _, _, _ = self.posterior_c[td](h, zp, cq, bs, ns)            
            cq_samples.append(cq)
            
            zp_mean, zp_logvar = self.prior_z[td](zp, cq, bs, ns)
            zpd = self.normal(zp_mean, zp_logvar)
            zp = zpd.sample()
            zp_samples.append(zp)
            
        zs = torch.cat(zp_samples, dim=-1)
        cs = torch.cat(cq_samples, dim=-1)
        xp = self.observation_decoder(zs, cs, bs, ns)
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def conditional_sample_mcmc_v1(self, x, mc=10):
        """
        Sample the model given a context X.
        Sample are obtained using an iterative appraoch.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        bs=x.shape[0]
        ns=x.shape[1]
        x = x.view(-1, 1, 28, 28)
        xp = self.conditional_sample_cq(x)["xp"]

        # convolutional encoder
        hc = self.encoder(x)
        hc = self.proj(hc)
        cq, _, _, _ = self.statistic(hc, bs, ns)

        # mcmc refinement
        for _ in range(mc):

            out = {l: [] for l in self.lst}
            out["cqs"] = [cq]
            out["h"] = [hc]
            
            hz = self.encoder(xp)
            hz = self.proj(hz)
            out["hz"] = [hz]

            # top-down stochastic inference - same direction generation
            out = self.top_down_stochastic_inference(out, bs, ns)

            # observation decoder p(x | z_{1:L}, c_{1:L})
            zs = torch.cat(out["zqs"], dim=1)
            cs = torch.cat(out["cqs"], dim=1)
            xp = self.observation_decoder(zs, cs, bs, ns)

        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def conditional_sample_mcmc_v2(self, x, mc=10, mode="use_q"):
        """
        Sample the model given a context X.
        Sample are obtained using an iterative appraoch.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        bs=x.shape[0]
        ns=x.shape[1]
        x = x.view(-1, 1, 28, 28)
        xp = self.conditional_sample_cq(x)["xp"]

        # convolutional encoder
        hc = self.encoder(x)
        hc = self.proj(hc)

        # mcmc refinement
        for _ in range(mc):

            out = {l: [] for l in self.lst}
            hz = self.encoder(xp)
            hz = self.proj(hz)
            out["hz"] = [hz]
            out["h"] = [hc]

            h = torch.cat([hc, hz], 1)
            cq_mean, cq_logvar, _, _ = self.statistic(h, bs, ns)
            cqd = self.normal(cq_mean, cq_logvar)
            cq = cqd.sample() 

            # top-down stochastic inference - same direction generation
            if mode == "use_q":
                out["cqs"] = [cq]
                out = self.top_down_stochastic_inference(out, bs, ns)
                zs = torch.cat(out["zqs"], dim=1)
                cs = torch.cat(out["cqs"], dim=1)

            elif mode == "use_p":
                zp_mean, zp_logvar = self.prior_z[-1](None, cq, bs, ns)
                zpd = self.normal(zp_mean, zp_logvar)
                zp = zpd.sample()
                zp_samples = [zp]
                cq_samples = [cq]
                for td in reversed(range(self.n_stochastic - 1)):
                    cq, _, _, _ = self.posterior_c[td](hz, zp, cq, bs, ns)            
                    cq_samples.append(cq)
                    
                    zp_mean, zp_logvar = self.prior_z[td](zp, cq, bs, ns)
                    zpd = self.normal(zp_mean, zp_logvar)
                    zp = zpd.sample()
                    zp_samples.append(zp)
                zs = torch.cat(zp_samples, dim=1)
                cs = torch.cat(cq_samples, dim=1)
            
            xp = self.observation_decoder(zs, cs, bs, ns)
        xp = xp.view(bs, ns, 1, 28, 28)
        return {"xp": xp}

    def compute_mll(self,
                    out: dict,
                    bs=None,
                    ns=None
                    ) -> dict:
        """
        Compute the variational lower-bound using 1 sample approximation of KLs.
        The result should be the same you obtain from loss before average.
        Recompute here to be sure that samples from the models make sense.

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
        kl_c = 0
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

        for l in range(self.n_stochastic):
            kl_c += (cqd[l].log_prob(cqs[l]) - cpd[l].log_prob(cqs[l])).sum(-1)
            kl_z += (zqd[l].log_prob(zqs[l]) - zpd[l].log_prob(zqs[l])).sum(-1)

        logpx = logpx.view(bs, -1)  # .sum(-1) / ns
        kl_z = kl_z.view(bs, -1)  # .sum(-1) / ns
        kl_c = kl_c.view(bs, -1).repeat(1, ns) / ns
        
        kl = kl_c + kl_z
        # Variational lower bound and weighted loss
        vlb = logpx - kl
        return {"vlb": vlb.squeeze()}

