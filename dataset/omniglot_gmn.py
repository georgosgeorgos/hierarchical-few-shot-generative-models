import gzip
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage.transform import rotate
from torch.utils import data

from dataset.base import BaseSetsDataset


class OmniglotSetsDatasetGMN(BaseSetsDataset):
    """
    Omniglot dataset used in Generative Matching Networks.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data(self):
        img = []
        if self.split in ["train_indistro", "val", "test_indistro"]:
            split = "train"
        else:
            split = self.split
        path = os.path.join(self.data_dir, self.dataset, split + "_small.npz")

        map_cls = {}
        file = np.load(path)
        for k in file:
            v = file[k]
            v = v / (2**self.n_bits - 1)
            v = 1 - v
            img.append(v.reshape(self.img_cls, -1))
            
        img = np.array(img).astype(np.float32)
        lbl = np.arange(img.shape[0]).reshape(-1, 1)
        lbl = lbl.repeat(self.img_cls, 1)
        return img, lbl, map_cls

    def augment_sets(self, sets, sets_lbl):
        """
        Augment training sets.
        """
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size, self.size, self.size)
        n_sets = len(augmented)

        n_cls = len(sets_lbl)
        augmented_lbl = np.arange(n_cls, 2*n_cls).reshape(-1, 1)
        augmented_lbl = augmented_lbl.repeat(self.sample_size, 1)
        
        # flip set
        for s in range(n_sets):
            flip_horizontal = np.random.choice([0, 1])
            flip_vertical = np.random.choice([0, 1])
            if flip_horizontal:
                augmented[s] = augmented[s, :, :, ::-1]
            if flip_vertical:
                augmented[s] = augmented[s, :, ::-1, :]

        # rotate images in set
        for s in range(n_sets):
            angle = np.random.uniform(0, 360)
            for item in range(self.sample_size):
                augmented[s, item] = rotate(augmented[s, item], angle)
        
        # even if the starting images are binarized, the augmented one are not
        augmented = np.expand_dims(augmented, 2)
        augmented = np.random.binomial(1, p=augmented, size=augmented.shape).astype(np.float32)
        augmented = np.concatenate([augmented, sets])
        augmented_lbl = np.concatenate([augmented_lbl, sets_lbl])

        perm = np.random.permutation(len(augmented))
        augmented = augmented[perm]
        augmented_lbl = augmented_lbl[perm]
        return augmented, augmented_lbl

class OmniglotSetsDatasetGMNRandom(OmniglotSetsDatasetGMN):
    """
    Omniglot dataset used in Generative Matching Networks.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data(self):
        img = []
        if self.split in ["train_indistro", "val", "test_indistro"]:
            split = "train"
        else:
            split = self.split
        path = os.path.join(self.data_dir, self.dataset, split + "_r.npy")

        map_cls = {}
        img = np.load(path)
        lbl = np.arange(img.shape[0]).reshape(-1, 1)
        lbl = lbl.repeat(self.img_cls, 1)
        return img, lbl, map_cls

