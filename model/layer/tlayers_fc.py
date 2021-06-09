import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from model.layer.layers_fc import PostPool, f_std, FCResBlock, Conv2d3x3


# class FiLM(nn.Module):
#     def __init__(self, dim):
#         super(FiLM, self).__init__()
#         self.conv1x1 = nn.Conv1d(dim, 2*dim, 1)
#         self.linear = nn.Linear(dim, 2*dim)

#     def forward(self, c):
#         bg = self.linear(c)
#         beta, gamma = bg.chunk(2, dim=1)
#         return beta, gamma


# class FiLMedResBlock(nn.Module):
#     """
#     Residual block.

#     Attributes:
#         dim: layer width.
#         num_layers: number of layers in residual block.
#         activation: specify activation.
#         batch_norm: use or not use batch norm.
#     """

#     def __init__(self,
#                  dim: int,
#                  num_layers: int,
#                  activation: nn,
#                  batch_norm: bool = False,
#                  dropout: bool = False
#                  ):
#         super(FiLMedResBlock, self).__init__()
#         self.num_layers = num_layers
#         self.activation = activation
#         self.batch_norm = batch_norm
#         self.dropout = dropout

#         self.block = []
#         for _ in range(self.num_layers):
#             layer = [nn.Linear(dim, dim)]
#             if self.batch_norm:
#                 layer.append(nn.BatchNorm1d(num_features=dim))
#             if self.dropout:
#                 layer.append(nn.Dropout(p=0.1))
#             self.block.append(nn.ModuleList(layer))
#         self.block = nn.ModuleList(self.block)

#     def forward(self, x, beta=1, gamma=0):
#         e = x + 0
#         for l, layer in enumerate(self.block):
#             for i in range(len(layer)):
#                 e = layer[i](e)
#                 e = beta * e + gamma
#             if l < (len(self.block) - 1):
#                 e = self.activation(e)
#                 e = beta * e + gamma
#         e = e + x
#         e = beta * e + gamma
#         return self.activation(e)

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

    def __init__(self,
                 hid_dim: int,
                 c_dim: int,
                 n_features: int,
                 activation: nn,
                 dropout_sample: bool = False,
                 mode: str="mean",
                 ):
        super(AttentiveStatistiC, self).__init__()

        self.c_dim = c_dim
        self.hid_dim = hid_dim
        self.activation = activation
        
        self.postpool = PostPool(self.hid_dim,
                                 self.c_dim,
                                 self.activation)
        self.aggregation_module = TBlock(self.hid_dim)

    def forward(self, h, bs, ns):
        # aggregate samples

        # X (bs, context, dim)
        h = h.view(bs, -1, self.hid_dim)
        z = h.mean(dim=1)

        a, att = self.aggregation_module(z, h)
        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, att

class TBlock(nn.Module):
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

    def __init__(self,
                 hid_dim,
                 n_head=4,
                 dropout=0.2,
                 ):
        super(TBlock, self).__init__()
        assert hid_dim % n_head == 0
        self.hid_dim = hid_dim
        self.n_head = n_head
        # layer norm before and after attention
        # self.ln1 = nn.BatchNorm1d(hid_dim)
        # self.ln2 = nn.BatchNorm1d(hid_dim)
        # self.ln1 = nn.LayerNorm(hid_dim)
        # self.ln2 = nn.LayerNorm(hid_dim)

        self.key = nn.Linear(hid_dim, hid_dim)
        self.query = nn.Linear(hid_dim, hid_dim)
        self.value = nn.Linear(hid_dim, hid_dim)

        # self.attn_drop = nn.Dropout(dropout)
        # self.resid_drop = nn.Dropout(dropout)

        # self.proj = nn.Linear(hid_dim, hid_dim)

    def mask_f(self, sim, flag=None):
        # mask sample of interest
        B, H, T, T = sim.size()

        if flag == "eye":
            eye = torch.eye(T).to(sim.device)
            mask = 1 - eye.view(1, 1, T, T).repeat(B, H, 1, 1)
        elif flag == "causal":
            one = torch.ones(T, T).to(sim.device)
            mask = torch.tril(one).view(1, 1, T, T)
        else:
            return sim
        sim = sim.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        return sim

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
        B, T, C = kv.size()
        #qq = qq.view(B, -1, C)
        
        # q (B, nh, T, hs)
        q = self.query(qq).view(B, -1, self.n_head, C //
                               self.n_head).transpose(1, 2)
        # k (B, nh, T, hs)
        k = self.key(kv).view(B, -1, self.n_head, C //
                             self.n_head).transpose(1, 2)
        # v (B, nh, T, hs)
        v = self.value(kv).view(B, -1, self.n_head, C //
                               self.n_head).transpose(1, 2)

        # sim (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        sim = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.shape[-1]))
        # select context for attention distribution
        # sim = self.mask_f(sim)
        # distribution over the context samples
        sim = F.softmax(sim, dim=-1)
        #sim = self.attn_drop(sim)

        # new values for memory
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = sim @ v
        # merge the heads
        # (B, T, nh*hs)
        out = out.transpose(1, 2).contiguous().view(qq.size())

        #x = self.ln1(qq + out)
        #x = self.resid_drop(x)
        #x = self.ln2(x + self.proj(x))
        return out, sim

#######################################################################
# Prior for z - p(z_l | z_{l+1}, c_l)
class AttentivePriorZ(nn.Module):
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
        super(AttentivePriorZ, self).__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.c_dim = c_dim
        self.z_dim = z_dim

        self.activation = activation
        self.drop = nn.Dropout(0.1)

        # modules
        self.linear_c = nn.Linear(self.c_dim, 2*self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.residual_block = FCResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.linear_params = nn.Linear(self.hid_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

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
            ez = z.view(-1, self.z_dim)
            ez = self.linear_z(ez)
            ez = ez.view(bs, -1, self.hid_dim)
        else:
            ez = c.new_zeros((bs, ns, self.hid_dim))

        if c is not None:
            ec = self.linear_c(c)
            gamma, beta = ec.chunk(2, dim=-1)
            gamma = gamma.view(bs, -1, self.hid_dim).expand_as(ez)
            beta = beta.view(bs, -1, self.hid_dim).expand_as(ez)
            #ec = ec.view(bs, -1, self.hid_dim).expand_as(ez)
        else:
            ec = ez.new_zeros((bs, ns, self.hid_dim))

        # sum and reshape
        e = gamma * ez + beta
        e = e.view(-1, self.hid_dim)
        e = self.activation(e)

        # for layer in self.linear_block:
        e = self.residual_block(e)

        # affine transformation to parameters
        e = self.linear_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)
        # e = self.drop(e)

        mean, logvar = e.chunk(2, dim=1)
        return mean, logvar

# #######################################################################
# INFERENCE NETWORK for z - q(z_l | z_{l+1}, c_l, h_l)
class AttentivePosteriorZ(nn.Module):
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
        super(AttentivePosteriorZ, self).__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.activation = activation
        self.drop = nn.Dropout(0.1)

        # modules
        self.linear_h = nn.Linear(self.hid_dim, self.hid_dim)
        self.linear_c = nn.Linear(self.c_dim, 2*self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.residual_block = FCResBlock(
            dim=self.hid_dim,
            num_layers=self.num_layers,
            activation=self.activation,
            batch_norm=True,
        )

        self.linear_params = nn.Linear(self.hid_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

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
        eh = h.view(-1, self.hid_dim)
        eh = self.linear_h(eh)
        eh = eh.view(bs, -1, self.hid_dim)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.linear_z(ez)
            ez = ez.view(bs, -1, self.hid_dim)
        else:
            ez = eh.new_zeros(eh.size())

        # embed c and expand for broadcast addition
        if c is not None:
            ec = self.linear_c(c)
            gamma, beta = ec.chunk(2, dim=-1)
            gamma = gamma.view(bs, -1, self.hid_dim).expand_as(ez)
            beta = beta.view(bs, -1, self.hid_dim).expand_as(ez)
            #ec = ec.view(bs, -1, self.hid_dim).expand_as(eh)
        else:
            ec = eh.new_zeros(eh.size())

        # sum and reshape
        e = gamma * (eh + ez) + beta
        e = e.view(-1, self.hid_dim)
        e = self.activation(e)

        # for layer in self.linear_block:
        e = self.residual_block(e)

        # affine transformation to parameters
        e = self.linear_params(e)
        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)
        
        mean, logvar = e.chunk(2, dim=1)
        return mean, logvar

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

    def __init__(self,
                 num_layers: int,
                 hid_dim: int,
                 c_dim: int,
                 z_dim: int,
                 h_dim: int,
                 activation: nn,
                 mode: str = "mean",
                 ladder=False
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

        self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.aggregation_module = TBlock(self.hid_dim)

        # self.residual_block = FCResBlock(
        #     dim=self.hid_dim,
        #     num_layers=self.num_layers,
        #     activation=self.activation,
        #     batch_norm=True,
        # )

        self.postpool = PostPool(self.hid_dim,
                                 self.c_dim,
                                 self.activation,
                                 self.ladder)

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
        ec = self.linear_c(c)
        ec = ec.view(bs, -1, self.hid_dim)
        
        if z is not None:
            ez = self.linear_z(z)
            ez = ez.view(bs, ns, self.hid_dim)
        else:
            ez = ec.new_zeros(ec.size())

        e = ez + ec.expand_as(ez)
        #e = e.view(-1, self.hid_dim)
        #e = self.activation(e)
        #e = self.residual_block(e)

        # map e to the moments
        e = e.view(bs, -1, self.hid_dim)
        z = e.mean(dim=1)
        a, att = self.aggregation_module(z, e)
       
        #a = a.unsqueeze(1) + ec
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

    def __init__(self,
                 num_layers: int,
                 hid_dim: int,
                 c_dim: int,
                 z_dim: int,
                 h_dim: int,
                 activation: nn,
                 mode: str = "mean",
                 ladder=False
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

        self.linear_h = nn.Linear(self.hid_dim, self.hid_dim)
        self.linear_c = nn.Linear(self.c_dim, self.hid_dim)
        self.linear_z = nn.Linear(self.z_dim, self.hid_dim)

        self.aggregation_module = TBlock(self.hid_dim)

        # self.residual_block = FCResBlock(
        #     dim=self.hid_dim,
        #     num_layers=self.num_layers,
        #     activation=self.activation,
        #     batch_norm=True,
        # )

        self.postpool = PostPool(self.hid_dim,
                                 self.c_dim,
                                 self.activation,
                                 self.ladder)

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
        eh = eh.view(bs, ns, -1)

        # map z to queries
        if z is not None:
            ez = self.linear_z(z)
            ez = ez.view(bs, ns, -1)
        else:
            ez = eh.new_zeros(eh.size())
        
        # map rc to keys and values
        ec = c.view(-1, self.c_dim)
        ec = self.linear_c(ec)
        ec = ec.view(bs, -1, self.hid_dim)

        # (b, sc, hdim)
        e = eh + ez + ec.expand_as(eh)
        #e = e.view(-1, self.hid_dim)
        #e = self.activation(e)
        #e = self.residual_block(e)

        e = e.view(bs, -1, self.hid_dim)
        z = e.mean(dim=1)
        a, att = self.aggregation_module(z, e)

        # map to moments
        if self.ladder:
            mean, logvar, feats = self.postpool(a)
            return mean, logvar, a, att, feats

        # map to moments
        mean, logvar = self.postpool(a)
        return mean, logvar, a, att, None

# class AttentiveObservationDecoder(nn.Module):
#     """
#     Parameterize (Bernoulli) observation model p(x | z, c).

#     Attributes:
#         batch_size: batch of datasets.
#         sample_size: number of samples in conditioning dataset.
#         num_layers: number of layers in residual block.
#         hid_dim: number of features for layer.
#         c_dim: number of features for latent summary.
#         z_dim: number of features for latent variable.
#         h_dim: number of features for the embedded samples.
#         n_stochastic_z: number of inference layers for z.
#         n_stochastic_c: number of inference layers for c.
#         activation: specify activation.
#     """

#     def __init__(self,
#                  num_layers: int,
#                  hid_dim: int,
#                  c_dim: int,
#                  z_dim: int,
#                  h_dim: int,
#                  n_stochastic_z: int,
#                  n_stochastic_c: int,
#                  activation: nn):
#         super(AttentiveDecoder, self).__init__()

#         self.num_layers = num_layers
#         self.hid_dim = hid_dim

#         self.c_dim = c_dim
#         self.z_dim = z_dim
#         self.h_dim = h_dim

#         self.n_stochastic_z = n_stochastic_z
#         self.n_stochastic_c = n_stochastic_c
#         self.activation = activation

#         self.linear_zs = nn.Linear(
#             self.n_stochastic_z * self.z_dim, self.hid_dim)
#         self.linear_cs = nn.Linear(
#             self.n_stochastic_c * self.c_dim, self.hid_dim)

#         self.linear_initial = nn.Linear(self.hid_dim, 256 * 4 * 4)

#         self.conv_layers = nn.ModuleList([
#             Conv2d3x3(256, 256),
#             Conv2d3x3(256, 256),
#             nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
#             Conv2d3x3(256, 128),
#             Conv2d3x3(128, 128),
#             nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
#             Conv2d3x3(128, 64),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
#         ])

#         self.bn_layers = nn.ModuleList([
#             nn.BatchNorm2d(256),
#             nn.BatchNorm2d(256),
#             nn.BatchNorm2d(256),
#             nn.BatchNorm2d(128),
#             nn.BatchNorm2d(128),
#             nn.BatchNorm2d(128),
#             nn.BatchNorm2d(64),
#             nn.BatchNorm2d(64),
#             nn.BatchNorm2d(64),
#         ])
#         self.memory = Memory(self.hid_dim)
#         self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, zs, cs, bs, ns):
#         """
#         Args:
#             zs: collection of all latent variables in all layers.
#             cs: collection of all latent variables for context in all layers.

#         Returns:
#             Moments of observation model.
#         """
#         # use z for all layers
#         ezs = self.linear_zs(zs)
#         ezs = ezs.view(bs, ns, -1)

#         if cs is not None:
#             ecs = self.linear_cs(cs)
#             ecs = ecs.view(bs, -1, self.hid_dim)#.expand_as(ezs)
#         else:
#             ecs = ezs.new_zeros(ezs.size())
#         # concatenate zs and c
#         e, _ = self.memory(ezs, ecs)
#         e = self.activation(e)
#         e = e.view(-1, self.hid_dim)

#         e = self.linear_initial(e)
#         e = e.view(-1, 256, 4, 4)

#         for conv, bn in zip(self.conv_layers, self.bn_layers):
#             e = conv(e)
#             e = bn(e)
#             e = self.activation(e)

#         e = self.conv_final(e)
#         # moments for the Bernoulli
#         p = torch.sigmoid(e)
#         return p


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

        statistics_network = StatisticNetwork(hid_dim,
                                              c_dim,
                                              h_dim,
                                              activation
                                              )
        h = torch.rand((batch_size, sample_size, hid_dim))
        print('embedding data size: {}'.format(h.size()))

        # q(c_L | D)
        mean, logvar, rcq = statistics_network.forward(h, bs, ns)
        cd = td.Normal(mean, f_std(logvar))
        c = cd.rsample()
        print('summary size: {}'.format(c.size()))

    test_layers()