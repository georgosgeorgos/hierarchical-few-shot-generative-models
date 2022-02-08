import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch.distributions as td
from torch import nn
import numpy as np

def normal_nll(x, mean, logvar):
    """Compute the negative log-likehood of x for a Gaussian."""
    return 0.5 * (
        + math.log(2*math.pi)
        + logvar
        + ((x - mean) ** 2) * torch.exp(-logvar)
    )

def normal_kl(mean0, logvar0, mean1, logvar1):
    """Compute the KL divergence between two Gaussians."""
    return 0.5 * (
        -1.0
        + logvar1
        - logvar0
        + torch.exp(logvar0 - logvar1)
        + ((mean0 - mean1) ** 2) * torch.exp(-logvar1)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, means, log_var, num_bits=8):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    log_scales = 0.5 * log_var
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    plus_in = inv_stdv * (centered_x + 1.0 / (2**num_bits-1))
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / (2**num_bits-1))
    cdf_min = approx_standard_normal_cdf(min_in)
    
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs

class DiscretizedNormalLikelihood(nn.Module):
    def __init__(self, continuous=False):
        super().__init__()
        self.continuous = continuous

    def forward(self, x, xp, continuous=False):
        loc, logvar = xp.chunk(2, dim=1)
        logvar = logvar.clamp(max=1)
        
        if continuous:
            px = self.normal(loc, logvar)
            logpx = px.log_prob(x)
            return logpx
        
        # for the discrete likelihood we want [-1, 1]
        x = x * 2 - 1
        return discretized_gaussian_log_likelihood(x=x, means=loc, log_var=logvar)

    def normal(self, loc, logvar):
        scale = torch.exp(logvar/2)
        distro = td.Normal(loc=loc, scale=scale)
        return distro

    def sample(self, xp):
        loc, logvar = xp.chunk(2, dim=1)
        px = self.normal(loc, logvar)
        img = px.sample()
        # sample = (sample + 1) / 2
        # sample = sample.clamp(min=0., max=1.)
        return img

