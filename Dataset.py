import os
from os import walk

import torchvision
from torch.utils.data import Dataset

import numpy as np


class SIDDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.noisy = []
        self.ground_truth = []
        self.transform = transform

        for (dirpath, dirnames, filenames) in walk(os.path.join(root_dir, 'train')):
            for filename in filenames:
                self.noisy.append((os.path.join(dirpath, filename), filename))

        for (dirpath, dirnames, filenames) in walk(os.path.join(root_dir, 'label')):
            for filename in filenames:
                self.ground_truth.append((os.path.join(dirpath, filename), filename))

        self.noisy = np.asarray(self.noisy)
        self.ground_truth = np.asarray(self.ground_truth)

    def __getitem__(self, item):
        train_path = self.noisy[item]
        ground_path = self.ground_truth[item]

        train = torchvision.io.read_image(train_path[0])
        label = torchvision.io.read_image(ground_path[0])

        if self.transform:
            train = self.transform(train)
            label = self.transform(label)

        return train, label

    def __len__(self):
        return len(self.noisy)