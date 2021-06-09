import argparse
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torchvision.utils import make_grid

sys.path.insert(0, os.path.join(str(Path.home()),
                                'hierarchical-few-shot-generative-models'))
from dataset import create_loader
from dataset.omniglot_ns import load_mnist_test_batch
from model import select_model
from utils.util import (load_args, load_checkpoint, model_kwargs, set_paths,
                        set_seed)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", default="NS", type=str, help="readable name for run",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./output",
    help="output directory for checkpoints and figures",
)
parser.add_argument(
    "--tag", type=str, default="", help="readable tag for interesting runs",
)
# dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="omniglot_ns",
)


def plot_samples(args, out_dict, dataset, sampling, name=None, img_dim=28, nc=1):
    ncols = 20
    nrows = args.batch_size
    import matplotlib.gridspec as gridspec
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    
    lst = [0 , 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(nrows):
        for j in range(args.sample_size_test):

            axes[i, j].imshow(out_dict[dataset]["xp"][lst[i]][j].view(-1, img_dim, img_dim).squeeze().cpu().numpy(), cmap="gray")
            axes[i, j+5].imshow(1 - out_dict[dataset]["x"][lst[i]][j].view(-1, img_dim, img_dim).squeeze().cpu().numpy(), cmap="gray")

            axes[i, j+10].imshow(out_dict[dataset]["c"][lst[i]][j].view(-1, img_dim, img_dim).squeeze().cpu().numpy(), cmap="gray")
            axes[i, j+15].imshow(1 - out_dict[dataset]["mcmc"][lst[i]][j].view(-1, img_dim, img_dim).squeeze().cpu().numpy(), cmap="gray")

    axes[0, 2].title.set_text('Reconstruction')
    axes[0, 7].title.set_text('Sets')
    axes[0, 12].title.set_text('Conditional')
    axes[0, 17].title.set_text('Refined')
    
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].axis("off")
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])

    # fig.tight_layout()
    if name is None:
        name = dataset + "_" + sampling
    fig.savefig("_img/sampling-" + name + ".png")
    fig.savefig("_img/sampling-" + name + ".pdf", bbox_inches='tight', dpi=300)


def test(args, model, test_loader, dataset, sampling, name):
    model.eval()
    batch = next(iter(test_loader))
    mnist_test_batch = load_mnist_test_batch(args)
    out_dict = {"omni": {}, "mnist": {}}
    with torch.no_grad():

        x_omni = batch
        x_omni = x_omni.to(args.device)
        x_omni = x_omni.bernoulli()
        #x_omni = x_omni.transpose(1, 0).contiguous()
        x_mnist = mnist_test_batch.to(args.device)
        x_mnist = x_mnist.bernoulli()

        out_dict["omni"]["x"] = x_omni
        out_dict["mnist"]["x"] = x_mnist

        out_dict["omni"]["xp"] = model.reconstruction(x_omni)
        out_dict["mnist"]["xp"] = model.reconstruction(x_mnist)

        if args.model in ["chfsgm", "cthfsgm"]:
            out_dict["omni"]["c"] = model.conditional_sample_cq(x_omni)["xp"]
            out_dict["mnist"]["c"] = model.conditional_sample_cq(x_mnist)["xp"]
        else:
            out_dict["omni"]["c"] = model.conditional_sample_cqL(x_omni)["xp"]
            out_dict["mnist"]["c"] = model.conditional_sample_cqL(x_mnist)["xp"]


        out_dict["omni"]["mcmc"] = model.conditional_sample_mcmc_v1(x_omni, 20)[
            "xp"]
        out_dict["mnist"]["mcmc"] = model.conditional_sample_mcmc_v1(x_mnist, 20)[
            "xp"]
    plot_samples(args, out_dict, dataset, sampling, name)


if __name__ == "__main__":
    s = 0
    s = set_seed(s)
    # parse args
    args = parser.parse_args()

    args.dataset = "omniglot_ns"

    args.name=""
    args.timestamp=""
    args.tag=""
    
    # experiment start time
    args = set_paths(args)
    args = load_args(args)
    
    args.batch_size = 10
    args.sample_size = 5
    args.sample_size_test = 5
    
    # set here
    nc = 1
    img_dim = 28
    _dataset = "omni"
    _sampling = "mcmc"
    _epochs = 400

    # dataloader
    args.dataset = "mnist"
    _, test_loader = create_loader(args, split="test", shuffle=False)
    # create model
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, _epochs, model)

    name = args.timestamp + "-" + args.tag + "-" + str(s)
    print(name)
    test(args, model, test_loader, _dataset, _sampling, name)
