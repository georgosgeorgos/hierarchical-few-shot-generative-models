import os

import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from model.cns import CNS
from model.layer.tlayers_conv import AttentiveStatistiC

import torch.distributions as td

# Model
class CTNS(CNS):
    """
    Neural Statistician with learnable aggregation over set.
    The learnable aggregation is implemented throught a variant of self-attention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # statistic network
        statistic_args = (
            self.num_layers,
            self.hidden_dim,
            self.c_dim,
            self.n_features,
            self.activation,
            False,
            self.aggregation_mode
        )
        self.statistic = AttentiveStatistiC(*statistic_args)

    def conditional_sample_refine_vis(self, x, mc=10):
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

        xp_lst = []

        #x = x.view(bs*ns, -1, self.img_dim, self.img_dim)
        xp = self.conditional_sample_cqL(x)["xp"]    

        xp_lst.append(xp.view(bs, ns, -1, self.img_dim, self.img_dim))
        # convolutional encoder
        hc = self.encoder(x)
        hc = self.proj(hc)
        cq, _, _, att = self.statistic(hc, bs, ns)

        for _ in range(mc):

            out={ l: [] for l in lst }
            out["cqs"] = [cq]

            hz = self.encoder(xp)
            hz = self.proj(hz)

            out = self.top_down_stochastic_inference(hz, out, bs, ns)    
            # observation decoder
            zs = torch.cat(out["zqs"], dim=1)
            xp = self.observation_decoder(zs, cq, bs, ns)
            xp = self.likelihood.sample(xp)
            
            xp_lst.append(xp.view(bs, ns, -1, self.img_dim, self.img_dim))

        xp = xp.view(bs, ns, -1, self.img_dim, self.img_dim)
        return {"xp": xp, "att": [att], "xp_lst": xp_lst}


    def conditional_sample_refine_vis_v2(self, x, mc=10, mode="use_p"):
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

        xp_lst = []

        #x = x.view(bs*ns, -1, self.img_dim, self.img_dim)
        xp = self.conditional_sample_cqL(x)["xp"]  

        xp_lst.append(xp.view(bs, ns, -1, self.img_dim, self.img_dim))  

        # convolutional encoder
        hc = self.encoder(x)
        hc = self.proj(hc)
         
        for _ in range(mc):

            out={ l: [] for l in lst }
            hz = self.encoder(xp)
            hz = self.proj(hz)

            _hc = hc.view(bs, ns, -1, 4, 4)
            _hz = hz.view(bs, ns, -1, 4, 4)
            #_hz = _hz[:, 0].unsqueeze(1)

            h = torch.cat([_hc, _hz], 1)
            h = h.view(bs*(2*ns), -1, 4, 4)
            # statistic network q(c_L | X)
            cq_mean, cq_logvar, _, att = self.statistic(h, bs, (2*ns))
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
            xp = self.likelihood.sample(xp)

            xp_lst.append(xp.view(bs, ns, -1, self.img_dim, self.img_dim))
            
        xp = xp.view(bs, ns, -1, self.img_dim, self.img_dim)
        return {"xp": xp, "att": [att], "xp_lst": xp_lst}
