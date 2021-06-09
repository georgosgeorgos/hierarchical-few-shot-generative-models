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

from dataset import create_loader
from dataset.omniglot_ns import load_mnist_test_batch
from model import select_model
from utils.util import (load_args, load_checkpoint, mkdirs, model_kwargs,
                        set_paths, set_seed)

sys.path.insert(0, os.path.join(str(Path.home()), 'hierarchical-few-shot-generative-models'))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=str,
    default="./output",
    help="output directory for checkpoints and figures",
)
# dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="omniglot_ns",
    help="select dataset",
)

def transfer_main(model, dataloader, S=1000):
    mll_test = []
    for batch in dataloader:
        # batchwise likelihoods
        batch_mll = []
        with torch.no_grad():
            #mnist_batch = load_mnist_test_batch(args)
            x = batch.to(args.device)
            # just to be sure that everything is binarized
            x = x.bernoulli()
            bs = x.shape[0]
            ns = x.shape[1]
            S = 1 
            for s in range(S):
                # if s % 100 == 0:
                #     print(s)
                out = model.forward(x)
                out = model.compute_mll(out)
                llh = out["vlb"].view(-1, 1)
                batch_mll.append(llh)
            # stack over importance samples
            tmp = torch.stack(batch_mll, dim=1)
            mll_test.extend(tmp)
    
    mll_test = torch.stack(mll_test, dim=0)
    mll_test = np.log(S) - torch.logsumexp(mll_test, dim=1)
    mll_test = mll_test.cpu().data.numpy()
    return mll_test

def main(args, epoch=400, split="test"):
    args.dataset="omniglot_ns"
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, epoch, model)
    model.eval()
    
    mll = None
    
    batch_size = args.batch_size
    print(batch_size)
    for context_size in [1, 2, 5, 10, 20]:

        print(context_size)
        args.sample_size_test = context_size
        args.sample_size = context_size
        # dataloader
        args.batch_size = (batch_size // context_size) * args.sample_size
        print(args.batch_size)
        args.split = split
        args.augment = False

        args.dataset="omniglot_ns"
        _, loader = create_loader(args, split="test", shuffle=False)
        
        mll_test = transfer_main(model, loader)
        print("mll_test", np.mean(mll_test), len(mll_test))
        print()

if __name__ == "__main__":
    s = 0
    s = set_seed(s)

    args = parser.parse_args()
    
    args.name=""
    args.timestamp=""
    args.tag=""
    
    args = set_paths(args)
    args = load_args(args)
    print(args)
    print()

    epoch = 400
    main(args, epoch)
