import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch.distributions as td
from torch import nn

class BernoulliLikelihood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, xp):
        px = td.Bernoulli(probs=xp)
        logpx = px.log_prob(x)
        return logpx

    def sample(self, xp, binary=False):
        px = td.Bernoulli(probs=xp)
        if binary:
            xp = px.sample()
        return xp

if __name__ == "__main__":
    distro=BernoulliLikelihood()
    x=torch.randn(10, 5, 3, 64, 64)
    xp=torch.zeros(10, 5, 3, 64, 64)

    logpx = distro(x, xp)
    print(logpx.size())
    print(distro.sample(xp).size())