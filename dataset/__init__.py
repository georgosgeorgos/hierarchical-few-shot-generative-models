import os

from torch.utils import data
from torchvision.transforms import Compose, Resize, ToTensor

from dataset.base import BaseSetsDataset
from dataset.mnist_binary import MNISTSetsDataset
from dataset.celeba import CelebaSetsDataset
from dataset.omniglot_gmn import OmniglotSetsDatasetGMN, OmniglotSetsDatasetGMNRandom
from dataset.omniglot_ns import OmniglotSetsDatasetNS
from dataset.util.transforms import DynamicBinarize, StaticBinarize

def select_dataset(args, split):
    if split == "vis":
        split = "test"
    kwargs = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "sample_size": args.sample_size,
        "num_classes_task": args.num_classes,
        "split": split,
        "augment": args.augment,
    }

    if args.dataset in ["cifar100", "cub", "minimagenet", "doublemnist", "triplemnist"]:
        dataset = BaseSetsDataset(**kwargs)
    
    elif args.dataset == "celeba":
        dataset = CelebaSetsDataset(**kwargs)

    elif args.dataset == "mnist":
        kwargs["binarize"] = True #args.binarize
        dataset = MNISTSetsDataset(**kwargs)
        
    elif args.dataset == "omniglot_back_eval":
        kwargs["binarize"] = True
        dataset = OmniglotSetsDatasetGMN(**kwargs)
    elif args.dataset == "omniglot_random":
        kwargs["binarize"] = True
        dataset = OmniglotSetsDatasetGMNRandom(**kwargs)

    # omniglot used in neural statistician
    elif args.dataset == "omniglot_ns":
        kwargs["binarize"] = True
        if split in ["test", "vis"]:
            kwargs["sample_size"] = args.sample_size_test
        if split == "vis":
            kwargs["binarize"] = False
            kwargs["split"] = "test"
        dataset = OmniglotSetsDatasetNS(**kwargs)

    else:
        print("No dataset available.")
    return dataset


def create_loader(args, split, shuffle, drop_last=False):
    dataset = select_dataset(args, split)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return dataset, loader


