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

sys.path.insert(
    0, os.path.join(str(Path.home()), "hierarchical-few-shot-generative-models")
)


from dataset import create_loader
from model import select_model
from utils.util import (
    dataset_kwargs,
    load_args,
    load_checkpoint,
    model_kwargs,
    set_paths,
    set_seed,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--output-dir",
    type=str,
    default="/output",
)


def p(_img, i, j, norm=False):
    img = _img[i][j]
    if norm:
        img = (img + 1) / 2
        img = (img - img.min()) / (img.max() - img.min())

    xhat = img.permute(1, 2, 0).squeeze()
    xhat = xhat.detach().cpu().numpy()
    xhat = np.minimum(np.maximum(0.0, xhat), 1.0)
    return xhat


def plot_samples(args, out, dataset, name=None, img_dim=28, nc=1):
    ncols = 20 + 3
    ns = args.sample_size_test
    nrows = args.batch_size

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7, 3),
        gridspec_kw=dict(
            width_ratios=[
                4,
                4,
                4,
                4,
                4,
                1,
                4,
                4,
                4,
                4,
                4,
                1,
                4,
                4,
                4,
                4,
                4,
                1,
                4,
                4,
                4,
                4,
                4,
            ],
            wspace=0.01,
            hspace=0.0,
        ),
    )

    lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # random.shuffle(lst)
    for i in range(nrows):
        for j in range(ns):
            axes[i, j].imshow(p(out[dataset]["xp"], lst[i], j))

            axes[i, j + 5 + 1].imshow(p(out[dataset]["x"], lst[i], j))

            axes[i, j + 10 + 2].imshow(p(out[dataset]["c"], lst[i], j))

            # axes[i, j + 15 + 3].imshow(p(out[dataset]["mcmc"], lst[i], j))

            axes[i, j + 15 + 3].imshow(p(out[dataset]["u"], lst[i], j))

    axes[0, 2].set_title("Reconstruction", fontsize=10)
    axes[0, 7 + 1].set_title("Set", fontsize=10)
    axes[0, 12 + 2].set_title("Conditional", fontsize=10)
    axes[0, 17 + 3].set_title("Unconditional", fontsize=10)
    # axes[0, 22 + 4].title.set_text("Unconditional")

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].axis("off")
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])

    # fig.tight_layout()
    if name is None:
        name = dataset + "_"
    fig.savefig("_img/sampling-" + name + ".png")
    fig.savefig("_img/sampling-" + name + ".pdf", bbox_inches="tight", dpi=1000)


def test(args, model, test_loader, dataset, name, img_dim=28, nc=1):
    ns = args.sample_size_test
    bs = args.batch_size
    out = {dataset: {}}
    with torch.no_grad():

        batch = next(iter(test_loader))
        x = batch.to(args.device)

        out[dataset]["x"] = x

        out[dataset]["xp"] = model.reconstruction(x)

        if args.model in ["chfsgm", "cthfsgm", "chfsgm_multiscale"]:
            out[dataset]["c"] = model.conditional_sample_cq(x)["xp"]
        else:
            out[dataset]["c"] = model.conditional_sample_cqL(x)["xp"]

        out[dataset]["mcmc"] = model.conditional_sample_mcmc_v1(x, 10)["xp"]
        # out[dataset]["mcmc"] = model.conditional_sample_mcmc_v1(x, 20)["xp"]
        out[dataset]["u"] = model.unconditional_sample(bs, ns)["xp"]

    plot_samples(args, out, dataset, name, img_dim, nc)


if __name__ == "__main__":
    s = 0
    s = set_seed(s)
    # parse args
    args = parser.parse_args()
    # choose dataset

    args.name = ""
    args.timestamp = ""
    args.tag = ""
    args.dataset = "celeba"

    # experiment start time
    args = set_paths(args)
    args = load_args(args)
    print(args)

    args.batch_size = 10
    args.sample_size = 5
    args.sample_size_test = 5
    # args.likelihood = "binary"

    # dataset
    dts = dataset_kwargs(args)
    nc = dts[args.dataset]["nc"]
    img_dim = dts[args.dataset]["size"]

    # dataloader
    _, test_loader = create_loader(args, split="train", shuffle=False)
    # create model
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    epochs = 800
    model = load_checkpoint(args, epochs, model)
    model.eval()

    name = args.timestamp + "-" + args.tag + "-" + str(s)
    print(name)
    test(args, model, test_loader, args.dataset, name, img_dim, nc)
