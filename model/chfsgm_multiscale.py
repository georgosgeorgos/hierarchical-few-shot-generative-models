import os
import sys

import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.chfsgm import CHFSGM
from model.layer.layers_multi_scale import (DecBlockC, DecBlockZ, Encoder,
                                            ObservationDec)


class CHFSGM_MULTISCALE(CHFSGM):
    """
    Hierarchical Few-Shot Generative Model.
    We implement a hierarchy over c.
    The joint prior can be written as:
    p(Z, c) = \prod_l p(Z_l | Z_{l+1}, c_l) p(c_l | c_{l+1}).
    """

    def process_string(self, string):
        lst = []
        string=string.strip().split(',')
        for s in string:
            s = s.split('-') 
            # (res, groups)
            lst.append((int(s[0]), int(s[1])))
        return lst

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # str_enc = '32-4, 16-4, 8-2, 4-2, 2-2, 1-2'
        # res_encoder = self.process_string(str_enc)

        self.res_encoder = self.process_string(self.str_enc)
        #[(32, 4), (16, 4), (8, 2), (4, 2), (2, 2), (1, 2)]
        
        self.res_gen_z = self.process_string(self.str_gen_z) #[(8, 2), (4, 2), (2, 2), (1, 2)]
        self.res_gen_c = self.process_string(self.str_gen_c) #[(8, 2), (4, 2), (2, 2), (1, 2)]

        # str_gen_z = '8-2, 4-2, 2-2, 1-2'
        # str_gen_c = '8-2, 4-2, 2-2, 1-2'
        
        self.res_gen_z.reverse()
        self.res_gen_c.reverse()

        # str_dec = '32-4, 16-4, 8-2'
        
        self.res_decoder = self.process_string(self.str_dec) #[(32, 4), (16, 4), (8, 2)]
        self.res_decoder.reverse()

        self.rz = [r for (r, g) in self.res_gen_z for _ in range(g)]
        self.rc = [r for (r, g) in self.res_gen_c for _ in range(g)]
        
        self.n_stochastic = sum([l[1] for l in self.res_gen_z])

        # convolutional encoder
        encoder_args = (self.hidden_dim, self.img_dim, self.in_ch, self.res_encoder)
        encoder = Encoder(*encoder_args) 
        
        # p(x | \bar z, \bar c)
        obs_args = (
            self.hidden_dim,
            self.c_dim,
            self.z_dim,
            self.img_dim,
            self.in_ch,
            self.ll,
            self.res_decoder,
            self.pixelcnn_mode,
        )

        self.encoder = Encoder(*encoder_args)

        latent_c = []
        latent_z = []

        # for resolution
        n_blocks = sum([k[1] for k in self.res_gen_z])
        for (r, g) in self.res_gen_z:
            # for group in resolution
            for _ in range(g):
                latent_z.append(
                    DecBlockZ(
                        self.hidden_dim,
                        self.z_dim,
                        self.c_dim,
                        res=r,
                        n_blocks=n_blocks,
                    )
                )

        n_blocks = sum([k[1] for k in self.res_gen_c])
        for (r, g) in self.res_gen_c:
            # for group in resolution
            for _ in range(g):
                latent_c.append(
                    DecBlockC(
                        self.hidden_dim,
                        self.z_dim,
                        self.c_dim,
                        res=r,
                        n_blocks=n_blocks,
                        aggregation_mode=self.aggregation_mode,
                    )
                )

        self.latent_c = nn.ModuleList(latent_c)
        self.latent_z = nn.ModuleList(latent_z)

        self.observation_model = ObservationDec(*obs_args)

        self.xc = nn.Parameter(torch.zeros(1, self.hidden_dim, 1, 1))
        self.xz = nn.Parameter(torch.zeros(1, self.hidden_dim, 1, 1))

    def forward(self, x: torch.Tensor, limit_layer=None, t=None) -> dict:
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

        lst = [
            "zqs",
            "zqd",  # posterior z
            "cqs",
            "cqd",  # posterior c
            "zpd",  # prior z
            "cpd",  # prior c
            "att",  # attention over context
            "h",
        ]
        out = {l: [] for l in lst}

        # convolutional encoder
        hh = self.encoder(x)
        out["h"] = hh
        
        # for resolution
        z = None
        c = None
        xc = self.xc
        xz = self.xz
        # forward prior/posterior for z and c
        _z = None
        _c = xc

        skip = False
        for i in range(self.n_stochastic):
            # for group in resolution
            hc = out["h"][self.rc[i]]
            hz = out["h"][self.rz[i]]

            if limit_layer is not None and i > limit_layer:
                skip = True
            xc, c, cqd, cpd = self.latent_c[i].forward(xc, hc, _z, ns, bs, skip, t)
            _c = xc
            xz, z, zqd, zpd = self.latent_z[i].forward(xz, hz, _c, ns, bs, skip, t)
            _z = xz
            
            out["cqd"].append(cqd)
            out["cqs"].append(c)
            out["cpd"].append(cpd)

            out["zqd"].append(zqd)
            out["zqs"].append(z)
            out["zpd"].append(zpd)

        xp = self.observation_model.forward(xz, xc, ns, bs)

        out["x"] = x.view(bs, ns, -1, self.img_dim, self.img_dim)
        out["xp"] = xp.view(bs, ns, -1, self.img_dim, self.img_dim)
        return out

    def reconstruction(self, x, t=None):
        bs = x.shape[0]
        ns = x.shape[1]
        xp = self.forward(x, t=t)["xp"]
        xp = xp.view(bs * ns, -1, self.img_dim, self.img_dim)
        rec = self.likelihood.sample(xp)
        rec = rec.view(bs, ns, -1, self.img_dim, self.img_dim)
        return rec

    def loss(
        self, out: dict, weight: float = 1.0, free_bits: bool = False
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

        x = out["x"].view(den, -1, self.img_dim, self.img_dim)
        xp = out["xp"].view(den, -1, self.img_dim, self.img_dim)

        # Likelihood for observation model
        logpx = self.likelihood(x, xp)
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
            if l == 0:
                # just for debugging
                kl_cL = td.kl_divergence(cqd[l], cpd[l]).sum()

            kl_c += td.kl_divergence(cqd[l], cpd[l]).sum()
            kl_z += td.kl_divergence(zqd[l], zpd[l]).sum()

        kl_c /= den
        kl_z /= den
        kl = kl_c + kl_z

        # Variational lower bound and weighted loss
        vlb = (logpx - kl)
        w_vlb = (weight * logpx) - (kl / weight)
        loss = - w_vlb

        return {
            "loss": loss,
            "vlb": vlb,
            "logpx": logpx,
            "kl_z": kl_z,
            "kl_cL": kl_cL / den,
            "kl_c": kl_c,
        }

    def unconditional_sample(self, bs=10, ns=5, limit_layer=None, t=None, device="cuda"):
        """
        Sample the model.

        Args:
            x: batch of datasets.
        Returns:
            Sample from the model.
        """
        # for resolution
        z = None
        c = None
        xc = self.xc
        xz = self.xz
        # forward prior/posterior for z and c
        _z = None
        _c = xc
        skip = False
        for i in range(self.n_stochastic):
            # for group in resolution
            if limit_layer is not None and i > limit_layer:
                skip = True
            xc, c, cpd = self.latent_c[i].sample_uncond(xc, _z, ns, bs, skip, t)
            _c = xc
            xz, z, zpd = self.latent_z[i].sample_uncond(xz, _c, ns, bs, skip, t)
            _z = xz

        xp = self.observation_model.forward(xz, xc, ns, bs)
        xp = self.likelihood.sample(xp)
        xp = xp.view(bs, ns, -1, self.img_dim, self.img_dim)
        return {"xp": xp}

    def conditional_sample_cq(self, x, limit_layer=None, t=None):
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

        # convolutional encoder
        hh = self.encoder(x)
        
        # for resolution
        z = None
        c = None
        xc = self.xc
        xz = self.xz
        # forward prior/posterior for z and c
        _z = None
        _c = xc
        skip = False
        for i in range(self.n_stochastic):
            # for group in resolution
            hc = hh[self.rc[i]]

            if limit_layer is not None and i > limit_layer:
                skip = True
            xc, c, cqd = self.latent_c[i].sample_cond(xc, hc, _z, ns, bs, skip, t)
            _c = xc
            xz, z, zpd = self.latent_z[i].sample_uncond(xz, _c, ns, bs, skip, t)
            _z = xz
            
        xp = self.observation_model.forward(xz, xc, ns, bs)
        xp = self.likelihood.sample(xp)
        xp = xp.view(bs, ns, -1, self.img_dim, self.img_dim)
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
        bs = x.shape[0]
        ns = x.shape[1]
        # x = x.view(bs*ns, -1, self.img_dim, self.img_dim)
        xp = self.conditional_sample_cq(x)["xp"]

        # convolutional encoder
        hhc = self.encoder(x)

        # mcmc refinement
        for _ in range(mc):

            # for resolution
            z = None
            c = None
            xc = self.xc
            xz = self.xz
            # forward prior/posterior for z and c
            _z = None
            _c = xc

            hhz = self.encoder(xp)
            for i in range(self.n_stochastic):
                # for group in resolution
                hc = hhc[self.rc[i]]
                hz = hhz[self.rz[i]]
                
                xc, c, cqd = self.latent_c[i].sample_cond(xc, hc, _z, ns, bs)
                _c = xc
                xz, z, zqd = self.latent_z[i].sample_cond(xz, hz, _c, ns, bs)
                _z = xz
            
            xp = self.observation_model.forward(xz, xc, ns, bs)
            xp = self.likelihood.sample(xp)
            xp = xp.view(bs, ns, -1, self.img_dim, self.img_dim)

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
        bs = x.shape[0]
        ns = x.shape[1]
        # x = x.view(bs*ns, -1, self.img_dim, self.img_dim)
        xp = self.conditional_sample_cq(x)["xp"]

        # convolutional encoder
        hhc = self.encoder(x)

        # mcmc refinement
        for _ in range(mc):
            z = None
            c = None
            xc = self.xc
            xz = self.xz
            # forward prior/posterior for z and c
            _z = None
            _c = xc

            hhz = self.encoder(xp)

            for i in range(self.n_stochastic):
                hc = hhc[self.rc[i]]
                hz = hhz[self.rz[i]]
            
                h = torch.cat([hc, hz], 1)
                
                # top-down stochastic inference - same direction generation
                if mode == "use_q":
                    xc, c, cqd = self.latent_c[i].sample_cond(xc, h, _z, ns, bs)
                    _c = xc
                    xz, z, zqd = self.latent_z[i].sample_cond(xz, h, _c, ns, bs)
                    _z = xz
                    

                elif mode == "use_p":
                    xc, c, cqd = self.latent_c[i].sample_cond(xc, h, _z, ns, bs)
                    _c = xc
                    xz, z, zpd = self.latent_z[i].sample_uncond(xz, _c, ns, bs)
                    _z = xz

            xp = self.observation_model.forward(xz, xc, ns, bs)
            xp = self.likelihood.sample(xp)
            xp = xp.view(bs, ns, -1, self.img_dim, self.img_dim)

        return {"xp": xp}

    def compute_mll(self, out: dict, bs=None, ns=None) -> dict:
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

        x = out["x"].view(bs * ns, -1, self.img_dim, self.img_dim)
        xp = out["xp"].view(bs * ns, -1, self.img_dim, self.img_dim)
        # Likelihood for observation model
        logpx = self.likelihood(x, xp)
        logpx = logpx.sum(-1).sum(-1).sum(-1)

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
            kl_c += (
                (cqd[l].log_prob(cqs[l]) - cpd[l].log_prob(cqs[l]))
                .sum(-1)
                .sum(-1)
                .sum(-1)
            )
            kl_z += (
                (zqd[l].log_prob(zqs[l]) - zpd[l].log_prob(zqs[l]))
                .sum(-1)
                .sum(-1)
                .sum(-1)
            )

        logpx = logpx.view(bs, -1).sum(-1)
        kl_z = kl_z.view(bs, -1).sum(-1)
        kl_c = kl_c.squeeze()
        
        kl = kl_c + kl_z
        # Variational lower bound and weighted loss
        vlb = logpx - kl
        return {"vlb": vlb.squeeze()}


