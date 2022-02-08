import numpy as np
import torch
import torch.distributions as td
from model.layer.utils_layer import get_1x1, get_3x3
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from model.layer.pixelcnn.model import PixelCNN


# log var = log sigma^2 = 2 log sigma
def f_std(logvar):
    return torch.exp(0.5 * logvar)


class ConvResBlock(nn.Module):
    def __init__(
        self,
        in_width,
        middle_width,
        out_width,
        down_rate=False,
        up_rate=False,
        residual=False,
        use_3x3=True,
        zero_last=False,
        dropout=False,
    ):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.residual = residual
        self.dropout = dropout

        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c3 = (
            get_3x3(middle_width, middle_width)
            if use_3x3
            else get_1x1(middle_width, middle_width)
        )
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

        if dropout:
            self.drop = nn.Dropout2d(p=0.1)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        if self.dropout:
            xhat = self.drop(xhat)
        out = x + xhat if self.residual else xhat
        if self.down_rate:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        if self.up_rate:
            out = F.interpolate(out, scale_factor=2)
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        nc: int,
        img_dim: int = 28,
        img_ch: int = 1,
        res=[(32, 1), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1)],
    ):
        super().__init__()
        self.img_dim = img_dim
        self.img_ch = img_ch

        self.in_conv = get_3x3(img_ch, nc)
        enc_blocks = []
        down_rate = False
        for i, (r, g) in enumerate(res):
            if r not in [res[0][0]]:
                down_rate = True
            for _ in range(g):
                use_3x3 = r > 2  # Don't use 3x3s for 1x1, 2x2 patches
                enc_blocks.append(
                    ConvResBlock(
                        nc,
                        nc // 4,
                        nc,
                        down_rate=down_rate,
                        residual=True,
                        use_3x3=use_3x3,
                    )
                )
                down_rate = False
        n_blocks = len(enc_blocks)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x.view(-1, self.img_ch, self.img_dim, self.img_dim)
        if self.img_dim == 28:
            x = F.interpolate(x, size=32)
        x = self.in_conv(x)
        h = {}
        h[x.shape[-1]] = x
        for block in self.enc_blocks:
            x = block(x)
            h[x.shape[-1]] = x
        return h


class DecBlockZ(nn.Module):
    def __init__(self, nc, zdim, cdim, n_blocks=3, res=32):
        super().__init__()
        self.res = res
        width = nc
        self.hid_dim = width
        use_3x3 = res > 2
        cond_width = nc // 4
        self.zdim = zdim
        self.cdim = cdim
        self.enc = ConvResBlock(
            width * 2, cond_width, zdim * 2, residual=False, use_3x3=use_3x3
        )
        self.prior = ConvResBlock(
            width,
            cond_width,
            zdim * 2 + width,
            residual=False,
            use_3x3=use_3x3,
            zero_last=True,
        )

        self.c_proj = get_1x1(width, width)
        self.c_proj.weight.data *= np.sqrt(1 / n_blocks)

        self.z_proj = get_1x1(zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = ConvResBlock(
            width, cond_width, width, residual=True, use_3x3=use_3x3
        )
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def get_moments(self, x, h, c, skip):
        if c is not None:
            x = x + c
        qm, qv = self.enc(torch.cat([x, h], dim=1)).chunk(2, dim=1)

        feats = self.prior(x)
        pm, pv, xpp = self.get_prior_moments(feats)
        if not skip:
            x = x + xpp
        return x, qm, qv, pm, pv

    def get_prior_moments(self, feats):
        pm, pv, xpp = (
            feats[:, : self.zdim, ...],
            feats[:, self.zdim : self.zdim * 2, ...],
            feats[:, self.zdim * 2 :, ...],
        )
        return pm, pv, xpp

    def sample_cond(self, x, h, c, ns, bs, skip=False, t=None):
        x = self.process_x(x, ns, bs)
        c = self.process_c(c, ns, bs)
        if c is not None:
            x = x + c
        qm, qv = self.enc(torch.cat([x, h], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = self.get_prior_moments(feats)
        if not skip:
            x = x + xpp

        zqd = self.normal(qm, qv, t)
        z = zqd.sample()

        if not skip:
            x = x + self.z_fn(z)
            x = self.resnet(x)
        return x, z, zqd

    def sample_uncond(self, x, c, ns, bs, skip=False, t=None):
        x = self.process_x(x, ns, bs)
        c = self.process_c(c, ns, bs)
        if c is not None:
            x = x + c
        feats = self.prior(x)
        pm, pv, xpp = self.get_prior_moments(feats)
        if not skip:
            x = x + xpp

        zpd = self.normal(pm, pv, t)
        z = zpd.sample()

        if not skip:
            x = x + self.z_fn(z)
            x = self.resnet(x)
        return x, z, zpd

    def normal(self, loc: torch.Tensor, log_var: torch.Tensor, t=None):
        log_std = log_var / 2
        # if t:
        #     log_std = log_std * t
        scale = torch.exp(log_std)
        distro = td.Normal(loc=loc, scale=scale)
        return distro

    def process_x(self, x, ns, bs):
        if x.shape[0] == 1:
            x = x.repeat(bs * ns, 1, 1, 1)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=self.res)
        return x

    def process_c(self, c, ns, bs):
        if c is not None:
            if c.shape[-1] != self.res:
                c = F.interpolate(c, size=self.res)
            c = c.view(-1, self.hid_dim, self.res, self.res)
            c = self.c_proj(c)
            c = c.unsqueeze(1)
            c = torch.repeat_interleave(c, ns, dim=1)
            c = c.view(-1, self.hid_dim, self.res, self.res)
        return c

    def forward(self, x, h, c, ns, bs, skip=False, t=None):
        x = self.process_x(x, ns, bs)
        c = self.process_c(c, ns, bs)

        x, qm, qv, pm, pv = self.get_moments(x, h, c, skip)

        zqd = self.normal(qm, qv, t)
        z = zqd.rsample()
        # input cq for dimensions
        zpd = self.normal(pm, pv, t)

        if not skip:
            x = x + self.z_fn(z)
            x = self.resnet(x)
        return x, z, zqd, zpd


class DecBlockC(nn.Module):
    def __init__(
        self, nc, zdim, cdim, n_blocks=3, res=32, aggregation_mode="mean", dropout=False
    ):
        super().__init__()
        self.res = res
        width = nc
        self.hid_dim = width
        use_3x3 = res > 2
        cond_width = nc // 4
        self.zdim = zdim
        self.cdim = cdim
        self.aggregation_mode = aggregation_mode

        self.enc = ConvResBlock(
            width,
            cond_width,
            cdim * 2,
            residual=False,
            use_3x3=use_3x3,
            dropout=dropout,
        )
        self.prior = ConvResBlock(
            width,
            cond_width,
            cdim * 2 + width,
            residual=False,
            use_3x3=use_3x3,
            zero_last=True,
            dropout=dropout,
        )

        self.z_proj = get_1x1(width, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)

        self.c_proj = get_1x1(cdim, width)
        self.c_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = ConvResBlock(
            width, cond_width, width, residual=True, use_3x3=use_3x3, dropout=dropout
        )
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.c_fn = lambda x: self.c_proj(x)

        if self.aggregation_mode == "lag":
            self.att = ConvLAG(nc, res)

    def get_moments(self, x, h, z, skip):
        ex = x.unsqueeze(1).expand_as(h) + h
        if z is not None:
            ex = ex + z
        ex = self.aggregation(ex)

        qm, qv = self.enc(ex).chunk(2, dim=1)

        feats = self.prior(x)
        pm, pv, xpp = self.get_prior_moments(feats)
        if not skip:
            x = x + xpp
        return x, qm, qv, pm, pv

    def get_prior_moments(self, feats):
        pm, pv, xpp = (
            feats[:, : self.cdim, ...],
            feats[:, self.cdim : self.cdim * 2, ...],
            feats[:, self.cdim * 2 :, ...],
        )
        return pm, pv, xpp

    def sample_uncond(self, x, z, ns, bs, skip=False, t=None):
        x = self.process_x(x, ns, bs)
        z = self.process_z(z, ns, bs)

        # if z is not None:
        #     x = x.unsqueeze(1).expand_as(z) + z
        #     print(x.size(), z.size())

        feats = self.prior(x)
        pm, pv, xpp = self.get_prior_moments(feats)
        if not skip:
            x = x + xpp

        cpd = self.normal(pm, pv, t)
        c = cpd.sample()

        if not skip:
            x = x + self.c_fn(c)
            x = self.resnet(x)
        return x, c, cpd

    def sample_cond(self, x, h, z, ns, bs, skip=False, t=None):
        x = self.process_x(x, ns, bs)
        h = self.process_h(h, ns, bs)
        z = self.process_z(z, ns, bs)

        ex = x.unsqueeze(1).expand_as(h) + h
        if z is not None:
            ex = ex + z
        ex = self.aggregation(ex)

        qm, qv = self.enc(ex).chunk(2, dim=1)

        feats = self.prior(x)
        pm, pv, xpp = self.get_prior_moments(feats)
        if not skip:
            x = x + xpp

        cqd = self.normal(qm, qv, t)
        c = cqd.sample()

        if not skip:
            x = x + self.c_fn(c)
            x = self.resnet(x)
        return x, c, cqd

    def aggregation(self, x):
        if self.aggregation_mode == "mean":
            x = x.mean(dim=1)
        elif self.aggregation_mode == "max":
            x = x.max(dim=1)[0]
        elif self.aggregation_mode == "lag":
            h = x.mean(dim=1)
            x, att = self.att(h, x)
        return x

    def normal(self, loc: torch.Tensor, log_var: torch.Tensor, t=None):
        log_std = log_var / 2
        # if t:
        #     log_std = log_std * t
        scale = torch.exp(log_std)
        distro = td.Normal(loc=loc, scale=scale)
        return distro

    def process_x(self, x, ns, bs):
        if x.shape[0] == 1:
            x = x.repeat(bs, 1, 1, 1)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=self.res)
        x = x.view(-1, self.hid_dim, self.res, self.res)
        return x

    def process_h(self, h, ns, bs):
        h = h.view(-1, ns, self.hid_dim, self.res, self.res)
        return h

    def process_z(self, z, ns, bs):
        if z is not None:
            if z.shape[-1] != self.res:
                z = F.interpolate(z, size=self.res)
            z = z.view(-1, self.hid_dim, self.res, self.res)
            z = self.z_proj(z)
            z = z.view(-1, ns, self.hid_dim, self.res, self.res)
        return z

    def forward(self, x, h, z, ns, bs, skip=False, t=None):
        x = self.process_x(x, ns, bs)
        h = self.process_h(h, ns, bs)
        # merge z from the previous layer with the context information
        z = self.process_z(z, ns, bs)

        x, qm, qv, pm, pv = self.get_moments(x, h, z, skip)

        cqd = self.normal(qm, qv, t)
        c = cqd.rsample()
        # input cq for dimensions
        cpd = self.normal(pm, pv, t)

        if not skip:
            x = x + self.c_fn(c)
            x = self.resnet(x)
        return x, c, cqd, cpd


class ConvLAG(nn.Module):
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
        nc,
        res,
        n_head=4,
        dropout=0.2,
    ):
        super(ConvLAG, self).__init__()
        assert nc % n_head == 0
        self.hid_dim = nc
        self.n_head = n_head
        self.res = res

        self.key = get_1x1(nc, nc)
        self.query = get_1x1(nc, nc)
        self.value = get_1x1(nc, nc)

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
        kv = kv.view(-1, self.hid_dim, self.res, self.res)

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
        return out, sim


#######################################################################

# Observation Decoder p(x| \bar z, \bar c)
class ObservationDec(nn.Module):
    """
    Parameterize observation model p(x | z, c).

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
        nc,
        zdim,
        cdim,
        img_dim: int = 28,
        img_ch: int = 1,
        likelihood: str = "binary",
        res=[(8, 1), (16, 1), (32, 1)],
        pixelcnn_mode=False,
    ):
        super(ObservationDec, self).__init__()

        self.img_dim = img_dim
        self.img_ch = img_ch
        self.ll = likelihood
        self.pixelcnn_mode = pixelcnn_mode

        self.hid_dim = nc
        self.z_conv = get_3x3(nc, nc)
        self.c_conv = get_3x3(nc, nc)
        enc_blocks = []
        up_rate = False
        for i, (r, g) in enumerate(res):
            if r not in [res[0][0]]:
                up_rate = True
            for _ in range(g):
                use_3x3 = r > 2  # Don't use 3x3s for 1x1, 2x2 patches
                enc_blocks.append(
                    ConvResBlock(
                        nc,
                        nc // 4,
                        nc,
                        up_rate=up_rate,
                        residual=True,
                        use_3x3=use_3x3,
                    )
                )
                up_rate = False
        n_blocks = len(enc_blocks)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

        if self.pixelcnn_mode:
            out_ch = nc // 4
            self.pixelcnn = PixelCNN(
                input_channels=out_ch, nr_filters=80, nr_softmax_bins=1
            )

        # binary
        out_ch = self.img_ch
        # normal
        if self.ll in ["discretized_normal", "normal"]:
            out_ch = 2 * out_ch
        # mixture of logistics
        elif self.ll == "discretized_mix_logistic":
            num_mixtures = 10
            out_ch = num_mixtures * 10

        if self.pixelcnn_mode:
            out_ch = nc // 4

        self.conv_final = get_1x1(nc, out_ch)
        self.conv_final.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, xz, xc, ns, bs):
        # res = xz.shape[-1]
        xz = self.z_conv(xz)
        if xc.shape[-1] != 1:
            xc = F.interpolate(xc, size=1)
        xc = self.c_conv(xc)
        xc = xc.unsqueeze(1)
        xc = torch.repeat_interleave(xc, ns, dim=1)
        xc = xc.view(-1, self.hid_dim, 1, 1)
        x = xc + xz
        for block in self.enc_blocks:
            x = block(x)

        if x.shape[-1] != self.img_dim:
            x = F.interpolate(x, size=self.img_dim)

        p = self.conv_final(x)

        if self.pixelcnn_mode:
            p = self.pixelcnn(p)
        # moments for the Bernoulli
        if self.ll == "binary":
            p = torch.sigmoid(p)
        return p
