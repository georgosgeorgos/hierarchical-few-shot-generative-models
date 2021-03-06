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
from torch.nn import functional as F

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


def transfer_main(model, dataloader, S=1000):
    mll_test = []

    for batch in dataloader:
        # batchwise likelihoods
        batch_mll = []
        with torch.no_grad():
            x = batch.to(args.device)
            # x=x.bernoulli()
            bs = x.shape[0]
            ns = x.shape[1]
            S = 1
            for s in range(S):
                out = model.forward(x)
                loss = model.loss(out)
                out = model.compute_mll(out)
                llh = out["vlb"].view(-1, 1)
                batch_mll.append(llh)
            # stack over importance samples
            tmp = torch.stack(batch_mll, dim=1)
            mll_test.extend(tmp)
            # print(len(mll))

    # stack over batch
    mll_test = torch.stack(mll_test, dim=0)
    const = mll_test.shape[1]
    mll_test = np.log(const) - torch.logsumexp(mll_test, dim=1)
    mll_test = mll_test.cpu().data.numpy()
    return mll_test


def main(args, epoch=400, split="test"):
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, epoch, model)
    model.eval()

    mll = None

    batch_size = args.batch_size
    print(batch_size)
    for context_size in [2, 5, 10, 20]:

        print(context_size)
        args.sample_size_test = context_size
        args.sample_size = context_size
        # dataloader
        args.batch_size = (batch_size // context_size) * args.sample_size
        print(args.batch_size)
        args.split = split
        args.augment = False

        args.dataset = "mnist"
        _, loader = create_loader(args, split="test", shuffle=False)

        mll_test = transfer_main(model, loader)
        print("mll_test", np.mean(mll_test) / args.sample_size, len(mll_test))
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

    print(args)
    print()

    main(args, epoch)
