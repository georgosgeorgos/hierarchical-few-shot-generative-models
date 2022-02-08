import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.nn import functional as F

sys.path.insert(
    0, os.path.join(str(Path.home()), "hierarchical-few-shot-generative-models")
)

from dataset import create_loader
from dataset.omniglot_ns import load_mnist_test_batch
from model import select_model
from utils.util import load_args, load_checkpoint, model_kwargs, set_paths, set_seed

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output-dir",
    type=str,
    default="/output",
    help="output directory for checkpoints and figures",
)

def plot_attention_hierarchy(args, out, dataset, name=None, nd=10):

    n = args.sample_size_test
    fig, axes = plt.subplots(
        nrows=11,
        ncols=(n + 3),
        figsize=(6, 4),
        gridspec_kw=dict(
            height_ratios=[4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4], hspace=0, wspace=0
        ),
    )
    # fig.subplots_adjust(wspace=0.01, hspace=0.01)
    digits = [k for k in range(nd)]

    valuel20 = out[dataset]["att"][0][:, 0, :, :].cpu().squeeze().numpy()
    valuel21 = out[dataset]["att"][0][:, 3, :, :].cpu().squeeze().numpy()

    xp2 = out[dataset]["xp_lst"][0]
    xp2 = p(xp2.cpu().squeeze().view(-1, n, 28, 28).numpy())

    valuel10 = out[dataset]["att"][1][:, 0, :, :].cpu().squeeze().numpy()
    valuel11 = out[dataset]["att"][1][:, 3, :, :].cpu().squeeze().numpy()

    xp1 = out[dataset]["xp_lst"][1]
    xp1 = p(xp1.cpu().squeeze().view(-1, n, 28, 28).numpy())

    valuel00 = out[dataset]["att"][2][:, 0, :, :].cpu().squeeze().numpy()
    valuel01 = out[dataset]["att"][2][:, 3, :, :].cpu().squeeze().numpy()

    xp0 = out[dataset]["xp_lst"][2]
    xp0 = p(xp0.cpu().squeeze().view(-1, n, 28, 28).numpy())

    x = 1 - out[dataset]["x"].cpu().squeeze().view(-1, n, 28, 28).numpy()

    print(valuel00.shape, valuel01.shape, valuel10.shape, valuel11.shape)
    print(valuel00)
    print(valuel10)
    print(valuel20)
    print(xp0.shape, xp1.shape)

    i = 0
    lst = [0, 4, 6]
    for ii in range(3):

        digit = digits[lst[ii]]

        d = 0
        axes[i, 0].imshow(xp2[digit][d], cmap="gray")
        axes[i, 1].bar(x=[k for k in range(n)], height=valuel20[digit])
        axes[i, 1].set_ylim([0, 1])

        axes[i, 2].bar(x=[k for k in range(n)], height=valuel21[digit])
        axes[i, 2].set_ylim([0, 1])

        axes[i + 1, 0].imshow(xp1[digit][d], cmap="gray")
        axes[i + 1, 1].bar(x=[k for k in range(n)], height=valuel10[digit])
        axes[i + 1, 1].set_ylim([0, 1])

        axes[i + 1, 2].bar(x=[k for k in range(n)], height=valuel11[digit])
        axes[i + 1, 2].set_ylim([0, 1])

        axes[i + 2, 0].imshow(xp0[digit][d], cmap="gray")
        axes[i + 2, 1].bar(x=[k for k in range(n)], height=valuel00[digit])
        axes[i + 2, 1].set_ylim([0, 1])

        axes[i + 2, 2].bar(x=[k for k in range(n)], height=valuel01[digit])
        axes[i + 2, 2].set_ylim([0, 1])

        for j in range(3, (n + 3)):
            axes[i, j].imshow(x[digit][j - 3], cmap="gray")
            axes[i + 1, j].imshow(x[digit][j - 3], cmap="gray")
            axes[i + 2, j].imshow(x[digit][j - 3], cmap="gray")

        i += 4

    for i in range(11):
        for j in range(n + 3):
            axes[i, j].axis("off")
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])

    fig.tight_layout()
    if name is None:
        name = dataset + "_" + sampling
    fig.savefig("_img/attention-h-" + name + ".png")
    fig.savefig("_img/attention-h-" + name + ".pdf", bbox_inches="tight", dpi=300)


def p(xp):
    return np.random.binomial(1, p=xp, size=xp.shape).astype(np.float32)


def plot_attention(args, out, dataset, name=None, nd=5):

    n = args.sample_size_test
    fig, axes = plt.subplots(nrows=nd, ncols=(n + 5), figsize=(6, 3))

    digits = [k for k in range(nd)]
    value0 = out[dataset]["att"][-1][:, 0, :, :].cpu().squeeze().numpy()
    value1 = out[dataset]["att"][-1][:, 1, :, :].cpu().squeeze().numpy()
    value2 = out[dataset]["att"][-1][:, 2, :, :].cpu().squeeze().numpy()
    value3 = out[dataset]["att"][-1][:, 3, :, :].cpu().squeeze().numpy()

    print(value0.shape)
    x = 1 - out[dataset]["x"].cpu().squeeze().view(-1, n, 28, 28).numpy()
    xp = out[dataset]["xp"].cpu().squeeze().view(-1, n, 28, 28).numpy()

    xp = p(xp)
    print(xp.shape)

    for i in range(nd):

        digit = digits[i]
        print(value0[digit])

        d = 0
        axes[i, 0].imshow(xp[digit][d], cmap="gray")

        axes[i, 1].bar(x=[k for k in range(n)], height=value0[digit])
        axes[i, 1].set_ylim([0, 1])

        axes[i, 2].bar(x=[k for k in range(n)], height=value1[digit])
        axes[i, 2].set_ylim([0, 1])

        axes[i, 3].bar(x=[k for k in range(n)], height=value2[digit])
        axes[i, 3].set_ylim([0, 1])

        axes[i, 4].bar(x=[k for k in range(n)], height=value3[digit])
        axes[i, 4].set_ylim([0, 1])

        for j in range(5, (n + 5)):
            axes[i, j].imshow(x[digit][j - 5], cmap="gray")

    for i in range(nd):
        for j in range(n + 5):
            axes[i, j].axis("off")
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    if name is None:
        name = dataset + "_" + sampling
    fig.savefig("_img/attention-" + name + ".png")
    fig.savefig("_img/attention-" + name + ".pdf", bbox_inches="tight", dpi=300)


def plot_no_attention(args, out, dataset, name=None, nd=10):

    n = args.sample_size_test
    fig, axes = plt.subplots(nrows=nd, ncols=(n + 3))

    digits = [k for k in range(nd)]
    # (bs, ns)
    value0 = [1 / n for _ in range(n)]
    value1 = [1 / n for _ in range(n)]

    x = 1 - out[dataset]["x"].cpu().squeeze().view(-1, n, 28, 28).numpy()
    xp = out[dataset]["xp"].cpu().squeeze().view(-1, n, 28, 28).numpy()

    xp = p(xp)
    for i in range(nd):

        digit = digits[i]

        d = 0
        axes[i, 0].imshow(xp[digit][d], cmap="gray")

        axes[i, 1].bar(x=[k for k in range(n)], height=value0)
        axes[i, 1].set_ylim([0, 1])

        axes[i, 2].bar(x=[k for k in range(n)], height=value1)
        axes[i, 2].set_ylim([0, 1])

        for j in range(3, (n + 3)):
            axes[i, j].imshow(x[digit][j - 3], cmap="gray")

    for i in range(nd):
        for j in range(n + 3):
            axes[i, j].axis("off")
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    if name is None:
        name = dataset + "_" + sampling
    fig.savefig("_img/no-attention-" + name + ".png")
    fig.savefig("_img/no-attention-" + name + ".pdf", bbox_inches="tight", dpi=300)


def test(args, model, test_loader, dataset, name):
    # batches for samples grid visualization
    batch = next(iter(test_loader))
    mnist_test_batch = load_mnist_test_batch(args)

    out = {"omni": {}, "mnist": {}}
    with torch.no_grad():
        x_omni = batch.to(args.device)
        # x_omni = x_omni.permute(1, 0, 2, 3, 4).contiguous()

        x_mnist = mnist_test_batch.to(args.device)
        # x_mnist = x_mnist.permute(1, 0, 2).contiguous()

        out[dataset]["x"] = x_omni

        if args.model in ["chfsgm", "cthfsgm"]:
            _out = model.conditional_sample_cq_hierarchy(x_omni)
        else:
            _out = model.conditional_sample_cqL(x_omni)

        out[dataset]["xp"] = _out["xp"]
        out[dataset]["att"] = _out["att"]
        # out[dataset]["xp_lst"] = _out["xp_lst"]

    plot_attention(args, out, dataset, name)
    # plot_no_attention(args, out, dataset, name)
    # plot_attention_hierarchy(args, out, dataset, name)


if __name__ == "__main__":
    s = 0
    s = set_seed(s)
    # args parser
    args = parser.parse_args()

    args.dataset = "omniglot_ns"

    # theia
    args.name = ""
    args.timestamp = ""
    args.tag = ""

    args = set_paths(args)
    args = load_args(args)
    print(args)

    args.batch_size = 10
    args.sample_size = 5  # + 5
    args.sample_size_test = 5  # + 5
    args.likelihood = "binary"

    # set here
    _dataset = "omni"
    _epochs = 600

    # dataloader
    _, test_loader = create_loader(args, split="test", shuffle=False)
    # create model

    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, _epochs, model)
    model.eval()

    name = args.timestamp + "-" + args.tag + "-" + str(s)
    print(name)
    test(args, model, test_loader, _dataset, name)
