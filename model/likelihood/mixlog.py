import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def log_discretized_logistic(x,
                             mean,
                             log_scale,
                             n_bins=256,
                             reduce='mean',
                             double=False):
    """
    Log of the probability mass of the values x under the logistic distribution
    with parameters mean and scale. The sum is taken over all dimensions except
    for the first one (assumed to be batch). Reduction is applied at the end.
    Assume input data to be inside (not at the edge) of n_bins equally-sized
    bins between 0 and 1. E.g. if n_bins=256 the 257 bin edges are:
    0, 1/256, ..., 255/256, 1.
    If values are at the left edge it's also ok, but let's be on the safe side
    Variance of logistic distribution is
        var = scale^2 * pi^2 / 3
    :param x: tensor with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param log_scale: tensor with log scale of distribution, shape has to be either
                  scalar or broadcastable
    :param n_bins: bin size (default: 256)
    :param reduce: reduction over batch: 'mean' | 'sum' | 'none'
    :param double: whether double precision should be used for computations
    :return:
    """
    log_scale = _input_check(x, mean, log_scale, reduce)
    if double:
        log_scale = log_scale.double()
        x = x.double()
        mean = mean.double()
        eps = 1e-14
    else:
        eps = 1e-7
    # scale = np.sqrt(3) * torch.exp(logvar / 2) / np.pi
    scale = log_scale.exp()

    # Set values to the left of each bin
    x = torch.floor(x * n_bins) / n_bins

    cdf_plus = torch.ones_like(x)
    idx = x < (n_bins - 1) / n_bins
    cdf_plus[idx] = torch.sigmoid(
        (x[idx] + 1 / n_bins - mean[idx]) / scale[idx])
    cdf_minus = torch.zeros_like(x)
    idx = x >= 1 / n_bins
    cdf_minus[idx] = torch.sigmoid((x[idx] - mean[idx]) / scale[idx])
    log_prob = torch.log(cdf_plus - cdf_minus + eps)
    log_prob = log_prob.sum((1, 2, 3))
    log_prob = _reduce(log_prob, reduce)
    if double:
        log_prob = log_prob.float()
    return log_prob


def discretized_mix_logistic_loss(x, l):
    """
    log-likelihood for mixture of discretized logistics, assumes the data
    has been rescaled to [-1,1] interval
    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp
    """

    # channels last
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)

    # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    xs = [int(y) for y in x.size()]
    # predicted distribution, e.g. (B,32,32,100)
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(
        xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + nn.Parameter(torch.zeros(xs + [nr_mix]).to(x.device),
                                       requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(
        xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(
              xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below
    # for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999,
    # log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which
    # never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero
    # instead of selecting: this requires use to use some ugly tricks to avoid
    # potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as
    # output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (
        1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + torch.log_softmax(logit_probs,
                                                                dim=-1)
    log_probs = torch.logsumexp(log_probs, dim=-1)

    return log_probs.unsqueeze(1)
    # loss_sep = -log_probs.sum((1, 2))  # keep batch dimension
    # return loss_sep


def _reduce(x, reduce):
    if reduce == 'mean':
        x = x.mean()
    elif reduce == 'sum':
        x = x.sum()
    return x


def _input_check(x, mean, scale_param, reduce):
    assert x.dim() == 4
    assert x.size() == mean.size()
    if scale_param.numel() == 1:
        scale_param = scale_param.view(1, 1, 1, 1)
    if reduce not in ['mean', 'sum', 'none']:
        msg = "unrecognized reduction method '{}'".format(reduce)
        raise RuntimeError(msg)
    return scale_param

def sample_from_discretized_mix_logistic(l):
    """
    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp
    """

    def to_one_hot(tensor, n):
        one_hot = torch.zeros(tensor.size() + (n,))
        one_hot = one_hot.to(tensor.device)
        one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), 1.)
        return one_hot

    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda:
        temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel,
                                       dim=4),
                             min=-7.)
    coeffs = torch.sum(torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel,
                       dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda:
        u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = nn.Parameter(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0,
                                 min=-1.),
                     max=1.)
    x2 = torch.clamp(torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 +
                                 coeffs[:, :, :, 2] * x1,
                                 min=-1.),
                     max=1.)

    out = torch.cat([
        x0.view(xs[:-1] + [1]),
        x1.view(xs[:-1] + [1]),
        x2.view(xs[:-1] + [1])
    ],
                    dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out

class DiscretizedMixtureLogisticLikelihood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, xp):
        # x = x.permute(0, 2, 3, 1)
        # xp = xp.permute(0, 2, 3, 1)
        # [-1, 1]
        x = x * 2 - 1
        
        if len(x.shape) == 5:
            bs, ns, c, h, w = x.size()
            x = x.view(bs*ns, -1, h, w)
            xp = xp.view(bs*ns, -1, h, w)
        return discretized_mix_logistic_loss(x=x, l=xp)

    def sample(self, xp):
        """
        Sample discretized mixture of logistic.
        xp (bs*ns, 10*mx, h, w)
        """
        img = sample_from_discretized_mix_logistic(l=xp)
        # rescale output [0, 1]
        img = (img + 1) / 2
        img = img.clamp(min=0., max=1.)
        return img

