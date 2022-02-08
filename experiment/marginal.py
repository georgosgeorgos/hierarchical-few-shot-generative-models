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
import torchvision

sys.path.insert(
    0, os.path.join(str(Path.home()), "hierarchical-few-shot-generative-models")
)

from dataset import create_loader
from dataset.omniglot_ns import load_mnist_test_batch
from model import select_model
from utils.util import (load_args, load_checkpoint, mkdirs, model_kwargs,
                        set_paths, set_seed)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=str,
    default="/output",
    help="output directory for checkpoints and figures",
)


def marginal_loglikelihood(model, dataloader, S=1000):
    mll_test = []

    for batch in dataloader:
        # batchwise likelihoods
        batch_mll = []  # store importance samples for batch x
        with torch.no_grad():
            x = batch
            x = x.to(args.device)
            bs = x.shape[0]
            ns = x.shape[1]
            S = 1
            for s in range(S):
                # if s % 100 == 0:
                #    print(s)
                # compute vlb
                out = model.forward(x, limit_layer=None)
                # loss = model.loss(out)
                out = model.compute_mll(out)
                # (batch, 1)
                llh = out["vlb"].view(-1, 1)
                # append loglikekihood for importance sample s
                batch_mll.append(llh)
            # stack over S importance samples
            tmp = torch.stack(batch_mll, dim=1)

            # store (batch, S) importance samples for batch x
            mll_test.extend(tmp)
            # print(len(mll))

    # stack over batch
    # (dataset_size, S)
    mll_test = torch.stack(mll_test, dim=0)
    # number of importance samples
    const = mll_test.shape[1]
    print(const)
    # logsumexp over importance samples
    mll_test = np.log(const) - torch.logsumexp(mll_test, dim=1)
    mll_test = mll_test.cpu().data.numpy()
    return mll_test


def init_folder(args):
    args.marginal_dir = os.path.join(
        args.output_dir, args.dataset, "eval", "marginal", args.name, args.ymd, args.hms
    )
    if args.tag != "":
        args.marginal_dir += "_" + args.tag
    mkdirs(args.marginal_dir)


def save_mll(args, mll, epoch):
    fn = args.name + "_" + args.timestamp + "_{}.json".format(args.sample_size_test)
    path = os.path.join(args.marginal_dir, fn)
    mll = np.mean(mll)
    mll_results = {
        "mll": float(mll),
        "dataset": args.dataset,
        "model": args.model,
        "timestamp": args.timestamp,
        "tag": args.tag,
        "split": args.split,
        "sample_size_test": args.sample_size_test,
        "sample_size_training": args.sample_size,
        "epoch": epoch,
    }
    with open(path, "w") as file:
        json.dump(mll_results, file)


def main(args, epoch=400, split="test"):
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, epoch, model)
    model.eval()

    mll = None
    init_folder(args)
    batch_size = args.batch_size

    for context_size in [2, 5, 10, 20]:

        args.sample_size_test = context_size
        args.sample_size = context_size
        # dataloader
        args.batch_size = int((batch_size / context_size) * args.sample_size)
        args.split = split
        args.augment = False
        _, loader_test = create_loader(args, split="test", shuffle=False)
        mll_test = marginal_loglikelihood(model, loader_test)

        print(context_size)
        print("mll_test", np.mean(mll_test) / args.sample_size, len(mll_test))
        # save_mll(args, mll_test, epoch)
        print()


if __name__ == "__main__":
    s = 0
    s = set_seed(s)
    args = parser.parse_args()

    args.likelihood = "binary"

    args.name = ""
    args.timestamp = ""
    args.tag = ""
    args.dataset = "omniglot_ns"
    epoch = 600

    args = set_paths(args)
    args = load_args(args)

    print()
    print(args)
    print()

    main(args, epoch)
