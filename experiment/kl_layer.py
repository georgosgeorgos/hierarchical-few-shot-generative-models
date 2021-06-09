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

sys.path.insert(0, os.path.join(str(Path.home()), 'hierarchical-few-shot-generative-models'))

import torch.distributions as td
from dataset import create_loader
from dataset.omniglot_ns import load_mnist_test_batch
from model import select_model
from utils.util import load_args, load_checkpoint, mkdirs, set_paths, set_seed, model_kwargs

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

def compute_kl_layers(model, dataloader):
    lst = ["kl_z", "kl_c"]
    log = {l: [] for l in []}

    kl_z_lst = []
    kl_c_lst = []
    
    for batch in dataloader:
        with torch.no_grad():
            x = batch
            x = x.to(args.device)
            out = model.forward(x)

            # posterior over c for all layers
            cqd = out["cqd"]
            # # prior over c for all layers
            cpd = out["cpd"]

            # posterior over z for all layers
            zqd = out["zqd"]
            # prior over z for all layers
            zpd = out["zpd"]

            bs = out["x"].shape[0]
            ns = out["x"].shape[1]
            den = bs * ns
            #print(bs)

            tmp_c = []
            tmp_z = []
            for l in range(len(out["zpd"])):
                
                tmp_c.append(td.kl_divergence(cqd[l], cpd[l]).sum() / den)
                tmp_z.append(td.kl_divergence(zqd[l], zpd[l]).sum() / den)

            kl_c_lst.append(torch.tensor(tmp_c).view(1, -1))
            kl_z_lst.append(torch.tensor(tmp_z).view(1, -1))

    kl_z_lst = torch.cat(kl_z_lst, dim=0)
    kl_c_lst = torch.cat(kl_c_lst, dim=0)
    print(kl_z_lst.mean(0))
    print(kl_z_lst.mean(0), kl_c_lst.mean(0))

def main(args, epoch=400, split="test"):
    args.likelihood = "binary"
    model = select_model(args)(**model_kwargs(args))
    model.to(args.device)
    model = load_checkpoint(args, epoch, model)
    model.eval()

    _, dataloader = create_loader(args, split="test", shuffle=False)
    kl_layer = compute_kl_layers(model, dataloader)   

if __name__ == "__main__":
    s = 0
    s = set_seed(s)

    args = parser.parse_args()

    args.name=""
    args.timestamp=""
    args.tag=""

    args = set_paths(args)
    args = load_args(args)
    print()
    print(args)
    print()

    epoch = 400
    main(args, epoch)