import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

import torch.distributions as td
from model.likelihood import select_likelihood


class BaseModel(nn.Module):
    """
    Attributes:
        ch_enc: base number of channel for the shared encoder.
        batch_size: batch of datasets.
        sample_size: number of samples in conditioning dataset.
        num_layers: number of layers in outidual block.
        hidden_dim: number of features for layer.
        hidden_dim_c: number of features for layer in context.
        c_dim: number of features for latent summary.
        z_dim: number of features for latent variable.
        h_dim: number of features for the embedded samples.
        n_stochastic: number of stochastic layers for generation and inference.
        activation: specify activation.
        print_parameters: print name and parameters for each layer.
    """
    def __init__(
        self,
        img_dim: int,
        in_ch: int,
        ch_enc: int,
        batch_size: int,
        sample_size: int,
        num_layers: int,
        hidden_dim: int,
        hidden_dim_c: int,
        c_dim: int,
        z_dim: int,
        h_dim: int,
        n_stochastic: int,
        activation: nn,
        dropout_sample: bool=False,
        print_parameters: bool=False,
        aggregation_mode: str="mean",
        ladder: bool=False,
        ll: str="binary"
    ):
        super(BaseModel, self).__init__()
        # dataset
        self.in_ch = in_ch
        self.img_dim = img_dim
        # output shared encoder
        self.ch_enc = ch_enc
        self.n_features = 4 * ch_enc * 4 * 4
        self.h_dim = self.n_features #h_dim
        # latent c
        self.c_dim = c_dim
        self.hidden_dim_c = hidden_dim_c
        # latent  z
        self.n_stochastic = n_stochastic
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        # network
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_sample = dropout_sample

        self.print_parameters = print_parameters

        self.aggregation_mode = aggregation_mode
        self.ladder = ladder
        self.ll = ll
        self.likelihood = select_likelihood(ll)()
        
        # initialize weights
        self.apply(self.weights_init)
        # print variables for sanity check and debugging
        self.print_model()

        self.lst = ["zqs", "zqd",  # posterior z
                    "cqs", "cqd",  # posterior c
                    "zpd",         # prior z
                    "cpd",         # prior c
                    "h",           # data
                    "att"          # attention over context
                    ]

    def print_model(self):
        if self.print_parameters:
            for i, pair in enumerate(self.named_parameters()):
                name, param = pair
                print("{} --> {}, {}".format(i + 1, name, param.size()))
            print()

    def count_params(self):
        nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number trainable parameters: ", nparams)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: batch of (small) datasets.
        Returns: collection of lists with priors for z and c, 
                 approximate posteriors for z and c, 
                 representations for X and x.
        """
        pass

    def normal(self,    
               loc: torch.Tensor, 
               log_var: torch.Tensor
               ):
        scale = torch.exp(log_var/2)
        distro = td.Normal(loc=loc, scale=scale)
        return distro

    def standard_normal(self, sample: torch.Tensor):
        n01 = td.Normal(
        loc=sample.new_zeros(sample.shape),
        scale=sample.new_ones(sample.shape)
        )
        return n01

    def loss(self, 
             out: dict, 
             weight: float=1.0,
             free_bits: bool = False
             ) -> (torch.Tensor, torch.Tensor):
        """
        Perform variational inference and compute the loss.

        Args:
            out: collection of lists with priors for z and c, 
                 approximate posteriors for z and c, 
                 representations for X and x.
            weight: reweights the lower-bound.
        Returns:
            Loss and variational lowerbound.
        """
        pass

    def step(self, 
             x: torch.Tensor, 
             alpha: float, 
             optimizer: nn, 
             clip_gradients: bool=True,
             free_bits: float=0.0
             ) -> dict:
        """
        Standard training step.

        Args:
            x: batch of datasets.
            alpha: hyp for weight.
            optimizer: optimizer.
            clip_gradients: clip gradients with thresh.

        Returns:
            dict with loss, vlb, logpx, kl.
        """

        assert self.training is True

        outputs = self.forward(x)
        loss_dict = self.loss(outputs, weight=(alpha + 1), free_bits=free_bits)
        loss = loss_dict["loss"]
        # perform gradient update
        optimizer.zero_grad()
        loss.backward()
        
        if clip_gradients:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)

        optimizer.step()
        # output loss, vlb, logpx, kl
        return loss_dict

    def unconditional_sample(self,):
        """
        Sample the model unconditionally.

        Args:
            ns: batch of datasets.
        Returns:
            Sample from the model.
        """
        pass
    
    def save(self, optimizer, path):
        torch.save(
            {
                "model_state": self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            path,
        )

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass
        