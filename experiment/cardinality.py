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

sys.path.insert(
    0, os.path.join(str(Path.home()), "hierarchical-few-shot-generative-models")
)
from dataset import create_loader
from dataset.omniglot_ns import load_mnist_test_batch
from model import select_model
from utils.util import (
    load_args,
    load_checkpoint,
    mkdirs,
    set_paths,
    model_kwargs,
    set_seed,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=str,
    default="/output",
    help="output directory for checkpoints and figures",
)



def f_std(logvar):
    return torch.exp(0.5 * logvar)


def cardinality(model, dataloader, dataloader_test, S=1):
    elbo = []
    elbo_train = []
    elbo_test = []

    kl_z = []
    kl_c = []
    p_x = []

    print(len(dataloader))
    for batch in dataloader:

        with torch.no_grad():
            x = batch.to(args.device)
            out = model.forward(x)
            _out = model.compute_mll(out)
            elbo.extend(_out["vlb"])
            elbo_train.extend(_out["vlb"])

    for batch in dataloader_test:

        with torch.no_grad():
            x = batch.to(args.device)
            out = model.forward(x)
            _out = model.compute_mll(out)
            _loss = model.loss(out)

            elbo.extend(_out["vlb"])
            elbo_test.extend(_out["vlb"])

            kl_c.append(_loss["kl_c"].item())
            kl_z.append(_loss["kl_z"].item())
            p_x.append(_loss["logpx"].item())

    print("kl", np.mean(kl_c), np.mean(kl_z))
    print("logpx", np.mean(p_x))

    elbo = torch.stack(elbo)
    elbo = elbo.cpu().data.numpy()

    elbo_train = torch.stack(elbo_train)
    elbo_train = elbo_train.cpu().data.numpy()

    elbo_test = torch.stack(elbo_test)
    elbo_test = elbo_test.cpu().data.numpy()
    return elbo, elbo_train, elbo_test


def main_loop(args, epoch=400, split="test"):
    # args.likelihood="binary"
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, epoch, model)
    model.eval()

    batch_size = args.batch_size
    print(args.batch_size)

    for context_size in [1, 2, 5, 10, 20]:
        print(context_size)

        args.sample_size_test = context_size
        args.sample_size = context_size

        # dataloader
        args.batch_size = 20  # (batch_size // context_size) * args.sample_size
        print(args.batch_size)
        args.split = split
        args.augment = False

        # args.dataset="omniglot_ns"
        args.dataset = "celeba"
        _, loader = create_loader(args, split="train", shuffle=False)
        _, loader_test = create_loader(args, split="test", shuffle=False)
        elbo, elbo_train, elbo_test = cardinality(model, loader, loader_test)

        print(context_size)
        print("elbo", np.mean(elbo))
        print("elbo_train", np.mean(elbo_train))
        print("elbo_test", np.mean(elbo_test))
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
