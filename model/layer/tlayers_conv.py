import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from model.layer.layers_conv import PostPool, f_std, ConvResBlock, Conv2d3x3, Conv2d1x1


class AttentiveStatistiC(nn.Module):
    """
    Compute the statistic q(c | X).
    Encode X, preprocess, aggregate, postprocess.
    Representation for context at layer L.

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        n_features: number of features in output of the shared encoder (256 * 4 * 4).
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        n_features: int,
        activation: nn,
        dropout_sample: bool = False,
        mode: str = "mean",
    ):
        super(AttentiveStatistiC, self).__init__()

        self.c_dim = c_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.activation = activation

        self.postpool = PostPool(
            self.num_layers, self.hid_dim, self.c_dim, self.activation
        )
        self.aggregation_module = ConvTBlock(self.hid_dim)

    def forward(self, h, bs, ns):
        # aggregate samples

        # X (bs, context, dim)
        h = h.view(bs, -1, self.hid_dim, 4, 4)
        z = h.mean(dim=1)

        a, att = self.aggregation_module(z, h)
        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, att


class ConvTBlock(nn.Module):
    """
    Transformer block for keys and values over c.
    rc = f(c_{l+1}, z_{l+1}).

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
    """

    def __init__(
        self,
        hid_dim,
        n_head=4,
        dropout=0.2,
    ):
        super(ConvTBlock, self).__init__()
        assert hid_dim % n_head == 0
        self.hid_dim = hid_dim
        self.n_head = n_head

        # layer norm before and after attention
        # self.ln1 = nn.BatchNorm1d(hid_dim)
        # self.ln2 = nn.BatchNorm1d(hid_dim)
        # self.ln1 = nn.LayerNorm(hid_dim)
        # self.ln2 = nn.LayerNorm(hid_dim)

        self.key = Conv2d1x1(hid_dim, hid_dim)
        self.query = Conv2d1x1(hid_dim, hid_dim)
        self.value = Conv2d1x1(hid_dim, hid_dim)

        # self.attn_drop = nn.Dropout(dropout)
        # self.resid_drop = nn.Dropout(dropout)

        # self.proj = nn.Linear(hid_dim, hid_dim)

    # def mask_f(self, sim, flag=None):
    #     # mask sample of interest
    #     B, H, T, T = sim.size()

    #     if flag == "eye":
    #         eye = torch.eye(T).to(sim.device)
    #         mask = 1 - eye.view(1, 1, T, T).repeat(B, H, 1, 1)
    #     elif flag == "causal":
    #         one = torch.ones(T, T).to(sim.device)
    #         mask = torch.tril(one).view(1, 1, T, T)
    #     else:
    #         return sim
    #     sim = sim.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
    #     return sim

    def forward(self, qq, kv, flag=None):
        """
        Compute attention weights alpha(k, q) and update representations.

        We compute a similarity measure between context samples and query sample.
        (k, v) are obtained from the context c_{l+1}. q from z_{l+1}.
        Compute attention weights with alpha(k, q) and update with alpha(k, q) v.

        Args:
            k: keys (batch_size, sample_size_q, sample_size_c, hid_dim)
            v: values (batch_size, sample_size_q, sample_size_c, hid_dim)
            q: query (batch_size, sample_size_q, hid_dim)
        Returns:
            Updated representations for the context memory at layer l.
        """
        B, T, C, W, H = kv.size()
        kv = kv.view(-1, self.hid_dim, 4, 4)

        # q (B, nh, T, c, h, w)
        q = (
            self.query(qq)
            .view(B, -1, self.n_head, C // self.n_head, W, H)
            .transpose(1, 2)
        )
        # k (B, nh, T, hs)
        k = (
            self.key(kv)
            .view(B, -1, self.n_head, C // self.n_head, W, H)
            .transpose(1, 2)
        )
        # v (B, nh, T, hs)
        v = (
            self.value(kv)
            .view(B, -1, self.n_head, C // self.n_head, W, H)
            .transpose(1, 2)
        )

        sim = torch.einsum("bnkcwh,bntcwh->bnkt", q, k)
        # sim (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # sim = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.shape[-1]))
        sim = sim * (1.0 / np.sqrt(W * H * C // self.n_head))
        # select context for attention distribution
        # sim = self.mask_f(sim)
        # distribution over the context samples
        sim = F.softmax(sim, dim=-1)
        # sim = self.attn_drop(sim)

        # new values for memory
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = torch.einsum("bnkt,bntcwh->bnkcwh", sim, v)
        # out = sim @ v
        # merge the heads
        # (B, T, nh*hs)
        out = out.transpose(1, 2).contiguous().view(qq.size())
        # x = self.ln1(qq + out)
        # x = self.resid_drop(x)
        # x = self.ln2(x + self.proj(x))
        return out, sim


#######################################################################


class AttentivePriorC(nn.Module):
    """
    Latent decoder for the context.

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        activation: nn,
        mode: str = "mean",
        ladder=False,
    ):
        super(AttentivePriorC, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.mode = mode

        self.ladder = ladder
        self.k = 2
        if ladder:
            self.k = 3

        self.c_dim = c_dim
        self.z_dim = z_dim

        self.activation = activation

        self.linear_c = Conv2d1x1(self.c_dim, self.hid_dim)
        self.linear_z = Conv2d1x1(self.z_dim, self.hid_dim)

        self.aggregation_module = ConvTBlock(self.hid_dim)

        # self.residual_block = FCResBlock(
        #     dim=self.hid_dim,
        #     num_layers=self.num_layers,
        #     activation=self.activation,
        #     batch_norm=True,
        # )

        self.postpool = PostPool(
            self.num_layers, self.hid_dim, self.c_dim, self.activation, self.ladder
        )

    def forward(self, z, c, bs, ns):
        """
        Combine z and rc using the attention mechanism and aggregating.
        Embed z if we have more than one stochastic layer.

        Args:
            z: latent for samples at layer l+1.
            rc: latent representations for context at layer l+1.
            (batch_size, sample_size, hid_dim)

        Returns:
            Moments of the generative distribution for c at layer l.
            Representations over context at layer l.
        """
        ec = c.view(-1, self.c_dim, 4, 4)
        ec = self.linear_c(ec)
        ec = ec.view(bs, -1, self.hid_dim, 4, 4)

        if z is not None:
            ez = z.view(-1, self.z_dim, 4, 4)
            ez = self.linear_z(ez)
            ez = ez.view(bs, ns, self.hid_dim, 4, 4)
        else:
            ez = ec.new_zeros(ec.size())

        e = ez + ec.expand_as(ez)
        # e = e.view(-1, self.hid_dim)
        # e = self.activation(e)
        # e = self.residual_block(e)

        # map e to the moments
        e = e.view(bs, -1, self.hid_dim, 4, 4)
        z = e.mean(dim=1)
        a, att = self.aggregation_module(z, e)

        # a = a.unsqueeze(1) + ec
        # map to moments
        if self.ladder:
            mean, logvar, feats = self.postpool(a)
            return mean, logvar, a, att, feats

        mean, logvar = self.postpool(a)
        return mean, logvar, a, att, None


#######################################################################


class AttentivePosteriorC(nn.Module):
    """
    Inference network q(c_l|c_{l+1}, z_{l+1}, H)
    gives approximate posterior over latent context.
    In this formulation there is no sharing of
    parameters between generative and inference model.

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        activation: specify activation.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        activation: nn,
        mode: str = "mean",
        ladder=False,
    ):
        super(AttentivePosteriorC, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.mode = mode
        self.ladder = ladder
        self.k = 2
        if ladder:
            self.k = 3

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.activation = activation

        self.linear_h = Conv2d1x1(self.hid_dim, self.hid_dim)
        self.linear_c = Conv2d1x1(self.c_dim, self.hid_dim)
        self.linear_z = Conv2d1x1(self.z_dim, self.hid_dim)

        self.aggregation_module = ConvTBlock(self.hid_dim)

        # self.residual_block = FCResBlock(
        #     dim=self.hid_dim,
        #     num_layers=self.num_layers,
        #     activation=self.activation,
        #     batch_norm=True,
        # )

        self.postpool = PostPool(
            self.num_layers, self.hid_dim, self.c_dim, self.activation, self.ladder
        )

    def forward(self, h, z, c, bs, ns):
        """
        Combine h, z, and c to parameterize the moments of the approximate
        posterior over c.
        Combine z and rc using the attention mechanism and aggregating.
        Embed z if we have more than one stochastic layer.

        Args:
            z: latent for samples at layer l+1.
            rc: latent representations for context at layer l+1. (batch_size, sample_size, hid_dim)
            h: embedding of the data at layer l.
            first: chek if first layer to match data dimension.

        Returns:
            Moments of the approximate posterior distribution for c at layer l.
            Representations over context at layer l.
        """
        eh = h.view(bs * ns, -1, 4, 4)
        eh = self.linear_h(h)
        eh = eh.view(bs, ns, -1, 4, 4)

        # map z to queries
        if z is not None:
            ez = z.view(-1, self.z_dim, 4, 4)
            ez = self.linear_z(ez)
            ez = ez.view(bs, ns, -1, 4, 4)
        else:
            ez = eh.new_zeros(eh.size())

        # map rc to keys and values
        ec = c.view(-1, self.c_dim, 4, 4)
        ec = self.linear_c(ec)
        ec = ec.view(bs, -1, self.hid_dim, 4, 4)

        # (b, sc, hdim)
        e = eh + ez + ec.expand_as(eh)
        # e = e.view(-1, self.hid_dim)
        # e = self.activation(e)
        # e = self.residual_block(e)

        e = e.view(bs, -1, self.hid_dim, 4, 4)
        z = e.mean(dim=1)
        a, att = self.aggregation_module(z, e)

        # map to moments
        if self.ladder:
            mean, logvar, feats = self.postpool(a)
            return mean, logvar, a, att, feats

        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, att, None
