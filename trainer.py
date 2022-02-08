import os
import shutil
import time

import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.omniglot_ns import load_mnist_test_batch
from utils.logger import Logger
from utils.util import (mkdirs, process_batch, save_args, save_checkpoint,
                   save_metrics, set_paths)
from utils.vis import vis_conditional_samples


def eval_model(args, model, loader, lst):
    log = {l: [] for l in lst}
    for batch in loader:
        with torch.no_grad():
            x = batch.to(args.device) #process_batch(args, batch, "test")["x"]
            out = model.forward(x)
            out = model.loss(out)
        
        for l in lst:
            log[l].append(out[l].data.item())
    return log

def lr_f(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run(args, model, optimizer, scheduler, loaders, lst):
    # experiment start time
    args.timestamp = time.strftime("%Y%m%d-%H%M%S")
    # set paths for checkpoints, figures, logs, tensorboards and wandb runs
    args = set_paths(args)
    # make directories if they don't exist
    mkdirs(args.ckpt_dir)
    mkdirs(args.fig_dir)
    mkdirs(args.log_dir)
    mkdirs(args.run_dir)


    train_loader, val_loader, test_loader = loaders
    args.len_tr = len(train_loader.dataset)
    args.len_vl = len(val_loader.dataset)
    args.len_ts = len(test_loader.dataset)

    print(len(train_loader))
    print(len(test_loader))
    # batches for samples grid visualization
    if args.is_vis:
        omni_test_batch = next(iter(vis_loader))
        mnist_test_batch = load_mnist_test_batch(args)
    
    # save configurations
    cfg = save_args(args)
    print(cfg)
    
    # copy script with model in ckpt directory
    fname = os.path.join(os.getcwd(), "model", args.model) + ".py"
    shutil.copy(fname, args.ckpt_dir)

    # set logger
    logger=Logger(args, lst, cfg)
    # init wand and tensorboard
    logger.init_writer()
    # log params and grads in wandb
    logger.log_grad(model) 

    # main training loop
    bar = tqdm(range(args.epochs))
    for epoch in bar:

        model.train()
        train_log = {l: [] for l in lst}

        # create new sets for training epoch
        train_loader.dataset.init_sets()
        for itr, batch in enumerate(train_loader):
            x = batch.to(args.device) #process_batch(args, batch, "train")["x"]
            out = model.step(x, 
                             args.alpha, 
                             optimizer, 
                             args.clip_gradients, 
                             args.free_bits)
            for l in lst:
                train_log[l].append(out[l].data.item())

        # reduce weight on loss
        args.alpha *= args.alpha_step
        
        model.eval()
        # eval model at each epoch
        val_log = eval_model(args, model, val_loader, lst)
        # test model at each epoch
        test_log = eval_model(args, model, test_loader, lst)

        # log metrics
        logger.add_logs(train_log, val_log, test_log, epoch)

        # check best vlb
        logger.update_best(val_log, test_log, epoch)

        # update learning rate if learning plateu
        if args.adjust_lr and args.dataset not in ["celeba"]:
            if args.scheduler == "plateau":
                scheduler.step(val_log[args.adjust_metric])
            else:
                scheduler.step()
        #update_learning_rate(args, optimizer, epoch)

        # print logs
        print_str = "VLB (tr:{:.2f}, ts:{:.2f}) KL_z (tr:{:.2f}, ts:{:.2f}) KL_c (tr:{:.2f}, ts:{:.2f}), alpha:{:.2f}, lr:{:.6f}"
        bar.set_description(print_str.format(train_log["vlb"], test_log["vlb"],
                                             train_log["kl_z"], test_log["kl_z"],
                                             train_log["kl_c"], test_log["kl_c"],
                                             args.alpha, lr_f(optimizer))
                                             )
        # save metrics at each epoch
        save_metrics(args, epoch, logger.get_metric())

        #visualize conditional sampling
        if (epoch + 1) % args.viz_interval == 0 and args.is_vis:
            if args.model != "vae":
                samples, samples_mnist = vis_conditional_samples(args, 
                                                                 epoch, 
                                                                 model, 
                                                                 omni_test_batch, 
                                                                 mnist_test_batch
                                                                 )
            # log samples
            #logger.add_sample(samples, samples_mnist)

        # checkpoint model at intervals
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(args, epoch, model, optimizer)

    logger.close_writer()
