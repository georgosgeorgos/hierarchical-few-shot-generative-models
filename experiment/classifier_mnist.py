import argparse
import json
import os
import pickle
import shutil
import string
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.distributions as td

sys.path.insert(
    0, os.path.join(str(Path.home()), "hierarchical-few-shot-generative-models")
)


from utils.util import (
    load_args,
    load_checkpoint,
    mkdirs,
    model_kwargs,
    set_paths,
    set_seed,
)
from model import select_model
from dataset.omniglot_ns import load_mnist_test_batch
from dataset import create_loader


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=str,
    default="/output",
    help="output directory for checkpoints and figures",
)


def classification_elbo(model, dataloader_train, dataloader_test):
    elbo = []
    elbo_test = []

    # fig, axes = plt.subplots(nrows=10, ncols=8, figsize=(8, 12),
    # gridspec_kw=dict(width_ratios=[8,4,4,4,4,4,2, 20], height_ratios=[6,6,6,6,6,6,6,6,6,6]))

    conditioning_sets = dataloader_train.dataset.make_sets_clsf(5)
    conditioning_elbo = {}
    for d in conditioning_sets:

        conditioning_sets[d] = conditioning_sets[d].bernoulli()
        print(d, conditioning_sets[d].size())
        x_cond = conditioning_sets[d].unsqueeze(0).to(args.device)
        out = model.forward(x_cond)
        _out = model.compute_mll(out)
        elbo_cond = _out["vlb"].sum()
        print(elbo_cond)
        conditioning_elbo[d] = elbo_cond

    predicted_class = []
    labels = []

    for i, batch in enumerate(dataloader_test):
        x, lbl = batch
        x = x.unsqueeze(1)
        x = x.bernoulli()
        # for digit in context set
        tmp = []
        print(i)

        x_new = x[0].squeeze().numpy()
        # axes[5, 0].imshow(x_new, cmap="gray")

        _i = 0
        dct_elbos = {"elbo_cond": [], "elbo_xx": [], "elbo_diff": []}
        for d in conditioning_sets:
            # x_cond = x_cond.to(args.device)
            # out = model.forward(x_cond)
            # _out = model.compute_mll(out)
            # elbo_cond = _out["vlb"].sum(1)
            # print(elbo_cond)

            with torch.no_grad():
                x = x.to(args.device)
                x_cond = conditioning_sets[d].unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)
                x_cond = x_cond.to(args.device)
                xx = torch.cat([x, x_cond], dim=1)

                # out = model.forward(x_cond)
                # _out = model.compute_mll(out)
                # elbo_cond = _out["vlb"].sum(1)
                # elbo_cond = elbo_cond

                out = model.forward(xx)
                _out = model.compute_mll(out)
                elbo_xx = _out["vlb"]

                diff = elbo_xx - conditioning_elbo[d]  # / elbo_cond
                tmp.append(diff)

        #     for j in range(5):
        #         _x_cond = x_cond[0][j].squeeze().cpu().numpy()
        #         axes[_i, j+1].imshow(1-_x_cond, cmap="gray")
        #     _i += 1

        #     dct_elbos["elbo_cond"].append(elbo_cond[0].squeeze().cpu().numpy())
        #     dct_elbos["elbo_xx"].append(elbo_xx[0].squeeze().cpu().numpy())
        #     dct_elbos["elbo_diff"].append(diff[0].squeeze().cpu().numpy())

        # digits = np.array([k for k in range(10)])
        # #for i in range(10):
        # axes[4, 7].plot(digits, np.array(dct_elbos["elbo_diff"]), "-o", label = "ELBO[x | X]")
        # axes[4, 7].legend(loc="lower right")
        # axes[5, 7].plot(digits, np.array(dct_elbos["elbo_xx"]), "v", label = "ELBO[X]", color='brown')
        # axes[5, 7].plot(digits, np.array(dct_elbos["elbo_cond"]), "*", label = "ELBO[x, X]", color='green')
        # axes[5, 7].legend(loc="upper right")

        # for i in range(10):
        #     for j in range(8):
        #         if i in [4, 5] and j in [7]:
        #             continue
        #         else:
        #             axes[i, j].axis("off")
        #             axes[i, j].set_yticklabels([])
        #             axes[i, j].set_xticklabels([])
        # plt.legend()
        # fig.savefig("_img/elbo-tmp.png")
        # break

        tmp = torch.stack(tmp, dim=-1)
        pred = tmp.max(dim=-1)[1]
        predicted_class.extend(pred)
        labels.extend(lbl)

    predicted_class = torch.stack(predicted_class)
    predicted_class = predicted_class.cpu().data.numpy()

    labels = torch.stack(labels)
    labels = labels.cpu().data.numpy()
    return predicted_class, labels


def classification_ll(model, dataloader_train, dataloader_test):

    distros = {}
    conditioning_sets = dataloader_train.dataset.make_sets_clsf(5)
    for d in conditioning_sets:

        conditioning_sets[d] = conditioning_sets[d].bernoulli()
        x_cond = conditioning_sets[d].unsqueeze(0)
        x_cond = x_cond.to(args.device)
        out = model.forward(x_cond)
        xp = out["xp"].repeat(args.batch_size, 1, 1, 1, 1)
        distros[d] = td.Bernoulli(probs=xp)

    predicted_class = []
    labels = []
    for i, batch in enumerate(dataloader_test):
        with torch.no_grad():

            x, lbl = batch
            # x = x.bernoulli()
            x = x.unsqueeze(1)
            x.repeat(1, x_cond.size(1), 1, 1, 1)
            x = x.to(args.device)

            # for digit in context set
            tmp = []
            for d in conditioning_sets:
                px = distros[d]
                logpx = px.log_prob(x)
                logpx = logpx.sum(-1).sum(-1).sum(-1)
                logpx = logpx.max(1)[0]

                tmp.append(logpx)

            tmp = torch.stack(tmp, dim=-1)
            pred = tmp.max(dim=-1)[1]
            predicted_class.extend(pred)
            labels.extend(lbl)

    predicted_class = torch.stack(predicted_class)
    predicted_class = predicted_class.cpu().data.numpy()

    labels = torch.stack(labels)
    labels = labels.cpu().data.numpy()
    return predicted_class, labels


def classification_kl(model, dataloader_train, dataloader_test):

    distros = {}
    conditioning_sets = dataloader_train.dataset.make_sets_clsf()
    for d in conditioning_sets:
        conditioning_sets[d] = conditioning_sets[d]  # .bernoulli()
        print(d, conditioning_sets[d].size())
        x_cond = conditioning_sets[d].unsqueeze(0)
        x_cond = x_cond.to(args.device)
        out = model.forward(x_cond)
        cqd = out["cqd"]
        distros[d] = cqd

    predicted_class = []
    labels = []
    for i, batch in enumerate(dataloader_test):
        with torch.no_grad():

            x, lbl = batch
            x = x.unsqueeze(1)
            # x = x.bernoulli()

            print(i)

            # for digit in context set
            tmp = []
            for d in conditioning_sets:
                x = x.to(args.device)

                _out = model.forward(x)
                cqd = _out["cqd"]

                kl = 0
                for k in range(len(cqd)):
                    _kl = td.kl_divergence(distros[d][k], cqd[k])
                    if len(_kl.shape) == 2:
                        _kl = _kl.sum(-1)
                    else:
                        _kl = _kl.sum(-1).sum(-1).sum(-1)
                    kl += _kl

                tmp.append(kl)

            tmp = torch.stack(tmp, dim=-1)
            pred = tmp.min(dim=-1)[1]
            predicted_class.extend(pred)
            labels.extend(lbl)

    predicted_class = torch.stack(predicted_class)
    predicted_class = predicted_class.cpu().data.numpy()

    labels = torch.stack(labels)
    labels = labels.cpu().data.numpy()
    return predicted_class, labels


def main_loop(args, epoch=400, split="test"):
    args.likelihood = "binary"
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, epoch, model)
    model.eval()

    args.sample_size = 5
    args.sample_size_test = 5

    # dataloader
    args.dataset = "mnist"
    _, loader_train = create_loader(args, split="train", shuffle=False)

    mnist = MNIST(
        root="/home/data/mnist_processed/",
        train=False,
        download=False,
        transform=ToTensor(),
    )
    loader_test = data.DataLoader(
        dataset=mnist,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    predicted_class, labels = classification_kl(model, loader_train, loader_test)

    print(predicted_class.shape, labels.shape)
    print((predicted_class == labels).mean())

    from sklearn.metrics import confusion_matrix

    C = confusion_matrix(labels, predicted_class)  # , normalize="pred")
    print(C)

    print(C.sum(1))
    print(C.sum(0))
    print()


if __name__ == "__main__":
    s = 0
    s = set_seed(s)

    args = parser.parse_args()

    args.dataset = "omniglot_ns"

    args.name = ""
    args.timestamp = ""
    args.tag = ""

    args = set_paths(args)
    args = load_args(args)

    print()
    print(args)
    print()

    epoch = 600
    main_loop(args, epoch)
