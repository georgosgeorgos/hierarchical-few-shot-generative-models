import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

# log var = log sigma^2 = 2 log sigma
def f_std(logvar):
    return torch.exp(0.5 * logvar)

class ConvResBlock(nn.Module):
    """
    Residual block.

    Attributes:
        dim: layer width.
        num_layers: number of layers in residual block.
        activation: specify activation.
        batch_norm: use or not use batch norm.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        activation: nn,
        batch_norm: bool = False,
        dropout: bool = False,
    ):
        super(ConvResBlock, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.block = []
        for _ in range(self.num_layers):
            layer = [Conv2d3x3(in_channels=dim, out_channels=dim)]
            if self.batch_norm:
                layer.append(nn.BatchNorm2d(num_features=dim))
            if self.dropout:
                layer.append(nn.Dropout(p=0.1))
            self.block.append(nn.ModuleList(layer))
        self.block = nn.ModuleList(self.block)

    def forward(self, x):
        e = x + 0
        for l, layer in enumerate(self.block):
            for i in range(len(layer)):
                e = layer[i](e)
            if l < (len(self.block) - 1):
                e = self.activation(e)
        # residual
        return self.activation(e + x)

class ResConv2d3x3(nn.Module):
    """
    Residual block.

    Attributes:
        dim: layer width.
        num_layers: number of layers in residual block.
        activation: specify activation.
        batch_norm: use or not use batch norm.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        activation: nn,
        batch_norm: bool = False,
    ):
        super(ResConv2d3x3, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.batch_norm = batch_norm
        
        self.block = []
        for _ in range(self.num_layers):
            layer = [Conv2d3x3(in_channels=dim, out_channels=dim)]
            if self.batch_norm:
                layer.append(nn.BatchNorm2d(num_features=dim))
            self.block.append(nn.ModuleList(layer))
        self.block = nn.ModuleList(self.block)

    def forward(self, x):
        e = x + 0
        for l, layer in enumerate(self.block):
            for i in range(len(layer)):
                e = layer[i](e)
            if l < (len(self.block) - 1):
                e = self.activation(e)
        # residual
        return self.activation(e + x)


class Conv2d3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, pad: int=1):
        super(Conv2d3x3, self).__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=pad, stride=stride
        )

    def forward(self, x):
        return self.conv(x)

class Conv2d1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv2d1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, stride=1
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()
        self.conv_pre=Conv2d1x1(in_channels, out_channels)
        self.conv_post=Conv2d1x1(out_channels, out_channels)
    def forward(x):
        x = self.conv_pre(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv_post(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.conv_pre=Conv2d1x1(in_channels, out_channels)
        self.conv_post=Conv2d1x1(out_channels, out_channels)
    def forward(x):
        x = self.conv_pre(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv_post(x)
        return x

class SharedConvEncoder(nn.Module):
    def __init__(self, nc: int, activation: nn, img_dim: int=28, img_ch: int=1):
        super(SharedConvEncoder, self).__init__()
        self.activation = activation
        self.img_dim = img_dim
        self.img_ch = img_ch
        
        self.conv_layers = nn.ModuleList(
            [
                Conv2d3x3(in_channels=self.img_ch, out_channels=nc),
                Conv2d3x3(in_channels=nc, out_channels=nc),
                Conv2d3x3(in_channels=nc, out_channels=nc, downsample=True),
                # (-1, 64, 14 , 14)
                Conv2d3x3(in_channels=nc, out_channels=2 * nc),
                Conv2d3x3(in_channels=2 * nc, out_channels=2 * nc),
                Conv2d3x3(in_channels=2 * nc, out_channels=2 * nc, downsample=True),
                # (-1, 128, 7, 7)
                Conv2d3x3(in_channels=2 * nc, out_channels=4 * nc),
                Conv2d3x3(in_channels=4 * nc, out_channels=4 * nc),
                Conv2d3x3(in_channels=4 * nc, out_channels=4 * nc, downsample=True)
                # (-1, 256, 4, 4)
            ]
        )

        self.bn_layers = nn.ModuleList(
            [
                nn.BatchNorm2d(num_features=nc),
                nn.BatchNorm2d(num_features=nc),
                nn.BatchNorm2d(num_features=nc),
                nn.BatchNorm2d(num_features=2 * nc),
                nn.BatchNorm2d(num_features=2 * nc),
                nn.BatchNorm2d(num_features=2 * nc),
                nn.BatchNorm2d(num_features=4 * nc),
                nn.BatchNorm2d(num_features=4 * nc),
                nn.BatchNorm2d(num_features=4 * nc),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.view(-1, self.img_ch, self.img_dim, self.img_dim)
        if self.img_dim == 64:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h)
            h = self.activation(h)
        return h


class PreProj(nn.Module):
    def __init__(self, hid_dim: int, n_features: int, activation: nn):
        super(PreProj, self).__init__()

        self.hid_dim = hid_dim
        self.n_features = n_features
        self.activation = activation

        # modules
        self.fc = Conv2d1x1(self.n_features, self.hid_dim)
        self.bn = nn.BatchNorm2d(self.hid_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # reshape and affine
        e = h.view(-1, self.n_features, 4, 4)
        e = self.fc(e)
        e = self.bn(e)
        e = self.activation(e)
        return e


class PostPool(nn.Module):
    def __init__(self, num_layers: int, hid_dim: int, c_dim: int, activation: nn, ladder=False):
        super(PostPool, self).__init__()

        self.c_dim = c_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.activation = activation
        self.drop = nn.Dropout(0.1)
        self.ladder = ladder

        self.k = 2
        if ladder: self.k = 3

        self.residual_block = ConvResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.linear_params = Conv2d1x1(self.hid_dim, self.k * self.c_dim)
        self.bn_params = nn.BatchNorm2d(self.k * self.c_dim)

    def forward(self, e):
        e = e.view(-1, self.hid_dim, 4, 4)
        e = self.residual_block(e)
        # affine transformation to parameters
        e = self.linear_params(e)
        e = self.bn_params(e)
        e = e.view(-1, self.k * self.c_dim, 4, 4)

        if self.ladder:
            mean, logvar, feats = e.chunk(self.k, dim=1)
            return mean, logvar, feats
        else:
            mean, logvar = e.chunk(self.k, dim=1)
            return mean, logvar

########################################################################
class StatistiC(nn.Module):
    """
    Compute the statistic q(c | X).
    Encode X, preprocess, aggregate, postprocess.
    Representation for context at layer L.

    Attributes:
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
        super(StatistiC, self).__init__()

        self.c_dim = c_dim
        self.mode = mode
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.activation = activation
        
        self.postpool = PostPool(self.num_layers, self.hid_dim, self.c_dim, self.activation)

    def forward(self, h, bs, ns):
        # aggregate samples
        h = h.view(bs, -1, self.hid_dim, 4, 4)
        if self.mode == "mean":
            a = h.mean(dim=1)
        elif self.mode == "max":
            a = h.max(dim=1)[0]
        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, None


########################################################################
# Prior for z - p(z_l | z_{l+1}, c_l)
class PriorZ(nn.Module):
    """
    Latent decoder for the sample.

    Attributes:
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
    ):
        super(PriorZ, self).__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.c_dim = c_dim
        self.z_dim = z_dim

        self.activation = activation
        self.drop = nn.Dropout(0.1)

        # modules
        self.linear_c = Conv2d1x1(self.c_dim, self.hid_dim)
        self.linear_z = Conv2d1x1(self.z_dim, self.hid_dim)

        self.residual_block = ConvResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.linear_params = Conv2d1x1(self.hid_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm2d( 2 * self.z_dim)

    def forward(self, z, c, bs, ns):
        """
        Args:
            z: latent for samples at layer l+1.
            c: latent representations for context at layer l.
        Returns:
            Moments of the generative distribution for z at layer l.
        """
        # combine z and c
        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim, 4, 4)
            ez = self.linear_z(ez)
            ez = ez.view(bs, -1, self.hid_dim, 4, 4)
        else:
            ez = c.new_zeros((bs, ns, self.hid_dim, 4, 4))

        if c is not None:
            c = c.view(-1, self.c_dim, 4, 4)
            ec = self.linear_c(c)
            ec = ec.view(bs, -1, self.hid_dim, 4, 4).expand_as(ez)
        else:
            ec = ez.new_zeros((bs, ns, self.hid_dim, 4, 4))

        # sum and reshape
        e = ez + ec
        e = e.view(-1, self.hid_dim, 4, 4)
        e = self.activation(e)

        # for layer in self.linear_block:
        e = self.residual_block(e)

        # affine transformation to parameters
        e = self.linear_params(e)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim, 4, 4)
        # e = self.drop(e)

        mean, logvar = e.chunk(2, dim=1)
        return mean, logvar


########################################################################
class EncoderBottomUp(nn.Module):
    """
    Bottom-up deterministic Inference network h_l = g(h_{l-1}, c) return representations
    for x=h_1 at each layer.

    Attributes:
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        activation: specify activation.

    """
    def __init__(self,
                 num_layers: int,
                 hid_dim: int,
                 activation: nn
                 ):
        super(EncoderBottomUp, self).__init__()
    
        self.hid_dim = hid_dim
        self.activation = activation
        self.num_layers = num_layers

        # modules
        self.residual_block = ConvResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward the deterministic path bottom-up h_{l} = f( g(h_{l-1}), c).
        Combine h_{l-1}, and c to compute h_l.

        Args:
            h: conditional embedding samples for layer h_{l-1}.
            c: latent variable for the context.
            first: if h is the data, map from h_dim to hid_dim.

        Returns:
            a new representation h_l for the data at layer l.
        """
        # reshape and affine
        e = h.view(-1, self.hid_dim, 4, 4)
        e = self.residual_block(e)
        return e

########################################################################
# INFERENCE NETWORK for z - q(z_l | z_{l+1}, c_l, h_l)
class PosteriorZ(nn.Module):
    """
    Inference network q(z|h, z, c) gives approximate posterior over latent variables.
    In this formulation there is no sharing of parameters between generative and inference model. 

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
    ):
        super(PosteriorZ, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.activation = activation
        self.drop = nn.Dropout(0.1)

        # modules
        self.linear_h = Conv2d1x1(self.hid_dim, self.hid_dim)
        self.linear_c = Conv2d1x1(self.c_dim, self.hid_dim)
        self.linear_z = Conv2d1x1(self.z_dim, self.hid_dim)

        self.residual_block = ConvResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.linear_params = Conv2d1x1(self.hid_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm2d(2*self.z_dim)

    def forward(self, h, z, c, bs, ns):
        """
        Args:
            h: conditional embedding for layer h_l.
            z: moments for p(z_{l-1} | z_l, c).
            c: context latent representation for layer l.
        Returns:
            moments for the approximate posterior over z 
            at layer l.
        """
        # combine h, z, and c
        # embed h
        eh = h.view(-1, self.hid_dim, 4, 4)
        eh = self.linear_h(eh)
        eh = eh.view(bs, -1, self.hid_dim, 4, 4)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim, 4, 4)
            ez = self.linear_z(ez)
            ez = ez.view(bs, -1, self.hid_dim, 4, 4)
        else:
            ez = eh.new_zeros(eh.size())

        # embed c and expand for broadcast addition
        if c is not None:
            c = c.view(-1, self.c_dim, 4, 4)
            ec = self.linear_c(c)
            ec = ec.view(bs, -1, self.hid_dim, 4, 4).expand_as(eh)
        else:
            ec = eh.new_zeros(eh.size())

        # sum and reshape
        e = eh + ez + ec
        e = e.view(-1, self.hid_dim, 4, 4)
        e = self.activation(e)

        # for layer in self.linear_block:
        e = self.residual_block(e)

        # affine transformation to parameters
        e = self.linear_params(e)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim, 4, 4)
        
        mean, logvar = e.chunk(2, dim=1)
        return mean, logvar


#######################################################################

class PriorC(nn.Module):
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
        ladder=False
    ):
        super(PriorC, self).__init__()

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

        # self.residual_block = ConvResBlock(
        #     dim=self.hid_dim,
        #     num_layers=self.num_layers,
        #     activation=self.activation,
        #     batch_norm=True,
        # )

        # self.linear_params = Conv2d1x1(self.hid_dim, 2 * self.c_dim)
        # self.bn_params = nn.BatchNorm2d(2*self.c_dim)

        self.postpool = PostPool(self.num_layers, self.hid_dim, self.c_dim, self.activation, self.ladder)

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
        c = c.view(-1, self.c_dim, 4, 4)
        ec = self.linear_c(c)
        ec = ec.view(bs, -1, self.hid_dim, 4, 4)

        if z is not None:
            ez = self.linear_z(z)
            ez = ez.view(bs, ns, self.hid_dim, 4, 4)
        else:
            ez = ec.new_zeros(ec.size())

        e = ez + ec.expand_as(ez)
        e = e.view(-1, self.hid_dim, 4, 4)
        #e = self.activation(e)
        #e = self.residual_block(e)

        # map e to the moments
        e = e.view(bs, -1, self.hid_dim, 4, 4)
        if self.mode == "mean":
            a = e.mean(dim=1)
        elif self.mode == "max":
            a = e.max(dim=1)[0]

        a = a.unsqueeze(1) + ec

        if self.ladder:
            mean, logvar, feats = self.postpool(a)
            return mean, logvar, a, None, feats

        mean, logvar = self.postpool(a)
        return mean, logvar, a, None, None


#######################################################################

class PosteriorC(nn.Module):
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
        ladder=False
    ):
        super(PosteriorC, self).__init__()

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

        # self.residual_block = ConvResBlock(
        #     dim=self.hid_dim,
        #     num_layers=self.num_layers,
        #     activation=self.activation,
        #     batch_norm=True,
        # )

        self.postpool = PostPool(self.num_layers, self.hid_dim, self.c_dim, self.activation, self.ladder)

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
        eh = self.linear_h(h)
        eh = eh.view(bs, ns, self.hid_dim, 4, 4)

        # map z to queries
        if z is not None:
            ez = self.linear_z(z)
            ez = ez.view(bs, ns, self.hid_dim, 4, 4)
        else:
            ez = eh.new_zeros(eh.size())

        # map rc to keys and values
        c = c.view(-1, self.c_dim, 4, 4)
        ec = self.linear_c(c)
        ec = ec.view(bs, -1, self.hid_dim, 4, 4)

        # (b, sc, hdim)
        e = eh + ez + ec.expand_as(eh)
        e = e.view(-1, self.hid_dim, 4, 4)
        # e = self.activation(e)
        # e = self.residual_block(e)
        
        e = e.view(bs, -1, self.hid_dim, 4, 4)
        if self.mode == "mean":
            a = e.mean(dim=1)
        elif self.mode == "max":
            a = e.max(dim=1)[0]
        a = a.unsqueeze(1) + ec
        
        # map to moments
        if self.ladder:
            mean, logvar, feats = self.postpool(a)
            return mean, logvar, a, None, feats

        mean, logvar = self.postpool(a)
        return mean, logvar, a, None, None


#######################################################################

# Observation Decoder p(x| \bar z, \bar c)
class ObservationDecoder(nn.Module):
    """
    Parameterize (Bernoulli) observation model p(x | z, c).

    Attributes:
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in residual block.
        hid_dim: number of features for layer.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        n_stochastic_z: number of inference layers for z.
        n_stochastic_c: number of inference layers for c.
        activation: specify activation.
    """

    def __init__(
        self,
        nc_dec: int,
        num_layers: int,
        hid_dim: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        n_stochastic_z: int,
        n_stochastic_c: int,
        activation: nn,
        img_dim: int = 28,
        img_ch: int=1,
        likelihood: str="binary"
    ):
        super(ObservationDecoder, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim

        nc = nc_dec
        self.nc_dec = nc_dec
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.img_dim = img_dim
        self.img_ch = img_ch
        
        self.n_stochastic_z = n_stochastic_z
        self.n_stochastic_c = n_stochastic_c
        self.activation = activation
        self.ll = likelihood

        self.linear_zs = Conv2d1x1(self.n_stochastic_z * self.z_dim, self.hid_dim)
        self.linear_cs = Conv2d1x1(self.n_stochastic_c * self.c_dim, self.hid_dim)

        self.post_proj = Conv2d1x1(self.hid_dim, self.nc_dec)

        pad=1
        if img_dim==28:
            pad=0

        self.conv_layers = nn.ModuleList(
            [
                Conv2d3x3(nc, nc),
                Conv2d3x3(nc, nc),
                nn.ConvTranspose2d(nc, nc, kernel_size=2, stride=2),
                Conv2d3x3(nc, nc//2),
                Conv2d3x3(nc//2, nc//2),
                nn.ConvTranspose2d(nc//2, nc//2, kernel_size=2, stride=2),
                Conv2d3x3(nc//2, nc//4),
                Conv2d3x3(nc//4, nc//4, pad=pad),
                nn.ConvTranspose2d(nc//4, nc//4, kernel_size=2, stride=2),
            ]
        )

        self.bn_layers = nn.ModuleList(
            [
                nn.BatchNorm2d(nc),
                nn.BatchNorm2d(nc),
                nn.BatchNorm2d(nc),
                nn.BatchNorm2d(nc//2),
                nn.BatchNorm2d(nc//2),
                nn.BatchNorm2d(nc//2),
                nn.BatchNorm2d(nc//4),
                nn.BatchNorm2d(nc//4),
                nn.BatchNorm2d(nc//4),
            ]
        )

        out_ch = self.img_ch
        if self.ll == "discretized_normal":
            out_ch = 2 * out_ch
        elif self.ll == "discretized_mix_logistic":
            # use mixture of logistics
            num_mixtures = 10
            out_ch = num_mixtures * 10    

        self.conv_final = Conv2d1x1(nc//4, out_ch)

    def forward(self, zs, cs, bs, ns):
        """
        Args:
            zs: collection of all latent variables in all layers.
            cs: collection of all latent variables for context in all layers.

        Returns:
            Moments of observation model.
        """
        # use z for all layers
        #print(zs.size(), cs.size())
        ezs = self.linear_zs(zs)
        ezs = ezs.view(bs, ns, self.hid_dim, 4, 4)

        if cs is not None:
            cs = cs.view(-1, self.n_stochastic_c*self.c_dim, 4, 4)
            ecs = self.linear_cs(cs)
            ecs = ecs.view(bs, -1, self.hid_dim, 4, 4).expand_as(ezs)
        else:
            ecs = ezs.new_zeros(ezs.size())

        # concatenate zs and c
        e = ezs + ecs
        e = self.activation(e)
        e = e.view(-1, self.hid_dim, 4, 4)

        e = self.post_proj(e)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = self.activation(e)
            
        if self.img_dim == 64:
            e = F.interpolate(e, scale_factor=2)
        
        p = self.conv_final(e)
        # moments for the Bernoulli
        if self.ll == "binary":
            p = torch.sigmoid(p)
        return p

if __name__ == "__main__":
    import torch.distributions as td

    def test_layers():
        c_dim = 3
        z_dim = 36
        h_dim = 100

        num_layers = 2
        hid_dim = 128
        batch_size = 10
        sample_size = 5

        bs = batch_size
        ns = sample_size

        n_stochastic_z = 3
        n_stochastic_c = 3

        activation = nn.ReLU()

        statistics_network = StatisticNetwork(hid_dim, c_dim, h_dim, activation)

        ladder_inference_network_z = LadderInferenceNetworkZ(
            num_layers, hid_dim, c_dim, z_dim, h_dim, activation
        )
        latent_decoder_z = LatentDecoderZ(
            num_layers, hid_dim, c_dim, z_dim, h_dim, activation
        )

        latent_decoder_c = LatentDecoderC(
            num_layers, hid_dim, c_dim, z_dim, h_dim, activation
        )
        inference_network_c = InferenceNetworkC(
            num_layers, hid_dim, c_dim, z_dim, h_dim, activation
        )

        observation_model = ObservationDecoder(
            num_layers,
            hid_dim,
            c_dim,
            z_dim,
            h_dim,
            n_stochastic_z,
            n_stochastic_c,
            activation,
        )

        h = torch.rand((batch_size, sample_size, hid_dim))
        print("embedding data size: {}".format(h.size()))

        # q(c_L | D)
        mean, logvar, rcq = statistics_network.forward(h, bs, ns)
        cd = td.Normal(mean, f_std(logvar))
        c = cd.rsample()
        print("summary size: {}".format(c.size()))

        # p(z_l | z_{l+1}, c) where z_{l+1} = None
        mean, logvar = latent_decoder_z.forward(None, c, bs, ns)
        zd = td.Normal(mean, f_std(logvar))
        z = zd.rsample()
        print("prior: {}".format(z.size()))

        # q(z_l | z_{l+1}, c, h) where z_{l+1} = None
        print("deterministic bottom-up inference: {}".format(h.size()))
        mean, logvar = ladder_inference_network_z.forward(h, None, c, bs, ns)
        zd = td.Normal(mean, f_std(logvar))
        z = zd.rsample()
        print("stochastic top-down inference: {}".format(z.size()))

        ################################# TEST p and q for context #################################
        mean, logvar, _ = latent_decoder_c.forward(z, c, bs, ns)
        print("mean pior over c: {}".format(mean.size()))
        rcp = torch.cat([mean, logvar], -1)
        mean, logvar, _ = inference_network_c.forward(h, z, rcp, bs, ns)
        print("mean posterior over c: {}".format(mean.size()))

        ################################ TEST ALL #################################################
        Z = []
        C = []

        h = torch.rand((batch_size, sample_size, hid_dim))

        # q(c_L | X)
        mean, logvar, _ = statistics_network.forward(h, bs, ns)
        cqd = td.Normal(mean, f_std(logvar))
        cq = cqd.rsample()
        C.append(cq)

        # genrative model
        # p(zp_L | c_L)
        m, lv = latent_decoder_z.forward(None, cq, bs, ns)
        zpd = td.Normal(m, f_std(lv))

        # inference model
        # q(zq_L | c_L, h)
        # h = ladder_inference_network_z.forward_deterministic(
        #     h, c, first=True, bs=bs, ns=ns)
        mean, logvar = ladder_inference_network_z.forward(h, None, c, bs, ns)
        zqd = td.Normal(mean, f_std(logvar))
        zq = zqd.rsample()
        Z.append(zq)

        cs = cq
        for l in range(2):
            print("layer: ", l)

            # p(cp_l | cp_{l+1}, zq_{l+1}})
            mean, logvar, _ = latent_decoder_c.forward(zq, cq, bs=bs, ns=ns)
            cpd = td.Normal(mean, f_std(logvar))
            rcp = torch.cat([mean, logvar], -1)
            # can you use the generative model also here zp?
            # q(cq_l | cq_{l+1}, zq_{l+1})
            cq, logvar, _, = inference_network_c.forward(h, zq, rcp, bs=bs, ns=ns)
            C.append(cq)

            # p(zp_l | zq_{l+1}, cq_l)
            m, lv = latent_decoder_z.forward(zq, cq, bs=bs, ns=ns)
            zpd = td.Normal(m, f_std(lv))
            zp = torch.cat([m, lv])

            # q(zq_l | zp_l, cq_l, h)
            # h = ladder_inference_network_z.forward_deterministic(
            #     h, cq, first=False, bs=bs, ns=ns)
            mean, logvar = ladder_inference_network_z.forward(h, zp, cq, bs=bs, ns=ns)
            zqd = td.Normal(mean, f_std(logvar))
            zq = zqd.rsample()
            Z.append(zq)

        zs = torch.cat(Z, dim=-1)
        zs = zs.view(bs, ns, -1)
        cs = torch.cat(C, dim=-1)
        cs = cs.view(bs, ns, -1)
        p = observation_model.forward(zs, cs, bs, ns)

        print(p.size())

    def test_memory():
        c_dim = 3
        z_dim = 36
        h_dim = 100

        num_layers = 2
        hid_dim = 32
        batch_size = 10
        sample_size_c = 5
        sample_size_z = 2

        n_stochastic = 3
        memory = MemoryC(hid_dim, c_dim, z_dim)

        k = torch.rand((batch_size, sample_size_c, h_dim))
        v = torch.rand((batch_size, sample_size_c, h_dim))
        q = torch.rand((batch_size, sample_size_z, h_dim))
        out = memory(k, v, q)

    #test_layers()
    img_dim=64
    encoder = SharedConvEncoder(64, nn.ReLU(), img_dim)
    x = torch.zeros(2, 5, 1, img_dim, img_dim)
    h = encoder(x)
    print(h.size())
    print()

    x = torch.zeros(2, 5, 256, 4, 4)
    x = x.view(-1, 256, 4, 4)
    conv = ConvResBlock(256, 3, nn.ReLU())

    z = conv(x)
    #print(z.size(), x.size())

    post = PostPool(3, 256, 32, nn.ReLU())
    mu, logvar = post(z)
    #print(mu.size(), logvar.size())

    prior_z = PriorZ(3, 256, 64, 32, 256, nn.ReLU())

    z = torch.zeros(10, 32, 4, 4)
    c = torch.zeros(10, 64, 4, 4)
    mu, logvar = prior_z(z, c, 2, 5)
    #print(mu.size(), logvar.size())

    posterior_z = PosteriorZ(3, 256, 64, 32, 256, nn.ReLU())

    z = torch.zeros(10, 32, 4, 4)
    c = torch.zeros(2, 64, 4, 4)
    h = torch.zeros(10, 256, 4, 4)
    mu, logvar = posterior_z(h, z, c, 2, 5)
    #print(mu.size(), logvar.size())

    posterior_c = PosteriorC(3, 256, 64, 32, 256, nn.ReLU())

    z = torch.zeros(10, 32, 4, 4)
    c = torch.zeros(2, 64, 4, 4)
    h = torch.zeros(10, 256, 4, 4)
    mu, logvar, _, _, _ = posterior_c(h, z, c, 2, 5)
    #print(mu.size(), logvar.size())

    statistic_c = StatistiC(3, 256, 64, 32, nn.ReLU())

    h = torch.zeros(10, 256, 4, 4)
    mu, logvar, _, _ = statistic_c(h, 2, 5)
    #print(mu.size(), logvar.size())

    prior_c = PriorC(3, 256, 64, 32, 256, nn.ReLU())

    z = torch.zeros(10, 32, 4, 4)
    c = torch.zeros(2, 64, 4, 4)
    h = torch.zeros(10, 256, 4, 4)
    mu, logvar, _, _, _ = prior_c(z, c, 2, 5)
    #print(mu.size(), logvar.size())

    zs = torch.zeros(10, 32*3, 4, 4)
    cs = torch.zeros(2, 64*3, 4, 4)
    obs = ObservationDecoder(256, 3, 256, 64, 32, 256, 3, 3, nn.ReLU(), img_dim)
    x = obs(zs, cs, 2, 5)
    print()
    print(x.size())