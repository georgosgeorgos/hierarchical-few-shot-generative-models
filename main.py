from utils.parser import parse_args

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils import data

from dataset import create_loader
from model import select_model
from trainer import run
from utils.util import select_optimizer, count_params, model_kwargs, set_seed

def main(args, lst):
    # load datasets and create loaders
    _, train_loader = create_loader(args, split="train", shuffle=True)
    _, val_loader = create_loader(args, split="val", shuffle=True)
    _, test_loader = create_loader(args, split="test", shuffle=False)

    _, vis_loader = create_loader(args, split="vis", shuffle=True)
    loaders = (train_loader, val_loader, test_loader, vis_loader)

    # create model
    model = select_model(args)(**model_kwargs(args))
    args.nparams = count_params(model)
    print(args.nparams)
    model.to(args.device)
    
    optimizer, scheduler = select_optimizer(args, model)

    if args.dry_run:
        from utils.trainer_dry import run
    else:
        from trainer import run
    run(args, model, optimizer, scheduler, loaders, lst)

if __name__ == "__main__":
    # log variables
    lst = ["loss", "vlb", "logpx", "kl_z", "kl_c"]
    args = parse_args()
    s = set_seed(args.seed)
    main(args, lst)
