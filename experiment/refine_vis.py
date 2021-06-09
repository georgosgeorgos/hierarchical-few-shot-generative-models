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
    ncols = 25
    nrows = 5
    import matplotlib.gridspec as gridspec
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    

    lst_i = [2, 3, 4, 5, 6]
    for i in range(nrows):
        for j in range(20):
            axes[i, j].imshow(1-out_dict[dataset]["refine_vis"][j][lst_i[i]][0].view(-1, img_dim, img_dim).squeeze().cpu().numpy(), cmap="gray")
        k = 0
        for j in range(20, 25):
            axes[i, j].imshow(out_dict[dataset]["x"][lst_i[i]][k].view(-1, img_dim, img_dim).squeeze().cpu().numpy(), cmap="gray")
            k += 1
                
    # axes[0, 2].title.set_text('Reconstruction')
    # axes[0, 7].title.set_text('Sets')
    # axes[0, 12].title.set_text('Conditional Samples')
    # axes[0, 17].title.set_text('Refined Samples')
    # axes[0, 22].title.set_text('Unconditional Samples')

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].axis("off")
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])

    # fig.tight_layout()
    if name is None:
        name = dataset + "_" + sampling
    fig.savefig("_img/sampling-refine-" + name + ".png")
    fig.savefig("_img/sampling-refine-" + name + ".pdf", bbox_inches='tight', dpi=300)


def test(args, model, test_loader, dataset, sampling, name):
    model.eval()
    batch = next(iter(test_loader))
    mnist_test_batch = load_mnist_test_batch(args)
    
    out_dict = {"omni": {}, "mnist": {}}
    with torch.no_grad():

        x_omni = batch.to(args.device)
        x_omni = x_omni.bernoulli()
        
        out_dict["omni"]["x"] = x_omni
        out_dict["omni"]["refine_vis"] = model.conditional_sample_refine_vis(x_omni, 20)["xp_lst"]
        out_dict["omni"]["refine_vis2_p"] = model.conditional_sample_refine_vis_v2(x_omni, 20, "use_p")["xp_lst"]
        out_dict["omni"]["refine_vis2_q"] = model.conditional_sample_refine_vis_v2(x_omni, 20, "use_q")["xp_lst"]

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
    _, test_loader = create_loader(args, split="test", shuffle=False)

    # create model
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, _epochs, model)

    name = args.timestamp + "-" + args.tag + "-" + str(s)
    print(name)
    test(args, model, test_loader, _dataset, _sampling, name)
