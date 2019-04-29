# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import utils

class NTU_FP_Dataset(torch.utils.data.Dataset):
    """ Dataset for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 relative=False,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.relative = relative
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        if self.relative:
            print("Relative mode")

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label, self.length, self.num_body = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:1]
            self.data = self.data[0:1]
            self.sample_name = self.sample_name[0:1]

        self.N, self.C, self.T, self.V, self.M = self.data.shape


    def __len__(self):
        return len(self.label)


    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index]) # [num channel, time, num joints, num perosn]

        length = self.length[index]
        data_numpy = data_numpy[:, 0:length, :, :]

        if self.relative:
            data_numpy = self.to_relative(data_numpy)
        else:
            # move to center
            means = data_numpy[:, 0:1, :, :].sum(axis=(2, 3), keepdims=True) / data_numpy.shape[2]
            data_numpy = data_numpy - means

        label = self.label[index]

        return data_numpy, label


    def to_relative(self, data):
        T = data.shape[1]
        for t in range(T):
            data[:, t, 1, :] = data[:, t, 1, :] - data[:, t, 0, :]
            data[:, t, 2, :] = data[:, t, 2, :] - data[:, t, 20, :]
            data[:, t, 3, :] = data[:, t, 3, :] - data[:, t, 2, :]
            data[:, t, 4, :] = data[:, t, 4, :] - data[:, t, 20, :]
            data[:, t, 5, :] = data[:, t, 5, :] - data[:, t, 4, :]
            data[:, t, 6, :] = data[:, t, 6, :] - data[:, t, 5, :]
            data[:, t, 7, :] = data[:, t, 7, :] - data[:, t, 6, :]
            data[:, t, 8, :] = data[:, t, 8, :] - data[:, t, 20, :]
            data[:, t, 9, :] = data[:, t, 9, :] - data[:, t, 8, :]
            data[:, t, 10, :] = data[:, t, 10, :] - data[:, t, 9, :]
            data[:, t, 11, :] = data[:, t, 11, :] - data[:, t, 10, :]
            data[:, t, 12, :] = data[:, t, 12, :] - data[:, t, 0, :]
            data[:, t, 13, :] = data[:, t, 13, :] - data[:, t, 12, :]
            data[:, t, 14, :] = data[:, t, 14, :] - data[:, t, 13, :]
            data[:, t, 15, :] = data[:, t, 15, :] - data[:, t, 14, :]
            data[:, t, 16, :] = data[:, t, 16, :] - data[:, t, 0, :]
            data[:, t, 17, :] = data[:, t, 17, :] - data[:, t, 16, :]
            data[:, t, 18, :] = data[:, t, 18, :] - data[:, t, 17, :]
            data[:, t, 19, :] = data[:, t, 19, :] - data[:, t, 18, :]
            data[:, t, 20, :] = data[:, t, 20, :] - data[:, t, 1, :]
            data[:, t, 21, :] = data[:, t, 21, :] - data[:, t, 7, :]
            data[:, t, 22, :] = data[:, t, 22, :] - data[:, t, 7, :]
            data[:, t, 23, :] = data[:, t, 23, :] - data[:, t, 11, :]
            data[:, t, 24, :] = data[:, t, 24, :] - data[:, t, 11, :]
        for t in range(T-1, 0, -1):
            data[:, t, 0, :] = data[:, t, 0, :] - data[:, t-1, 0, :]
        data[:, 0, 0, :] = data[:, 0, 0, :] - data[:, 0, 0, :]
        return data


