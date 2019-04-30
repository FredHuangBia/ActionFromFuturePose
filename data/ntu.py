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
from matplotlib import pyplot as plt

# operation
from . import utils

class NTU_Dataset(torch.utils.data.Dataset):
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
                 img_like=False,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.img_like = img_like

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index]) # [1, num channel, time, num joints, num perosn]
        label = self.label[index]
        
        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        if self.img_like:
            # (3, 300, 25, M)
            H = 36
            W = 64
            delta_t = 5
            T = int(300 / delta_t)
            M = data_numpy.shape[3]
            data = np.zeros([25, T, H, W], dtype=np.float32)
            for m in range(M):
                for t in range(0, 300, delta_t):
                    for p in range(25):
                        h_dec = data_numpy[1, int(t/delta_t), p, m]
                        w_dec = data_numpy[0, int(t/delta_t), p, m]
                        if (h_dec == 0 and w_dec == 0):
                            continue
                        h_dec = (h_dec + 1) / 2
                        w_dec = (w_dec + 1) / 2
                        h = max(0, min(H-1, H - int(round(H*h_dec))))
                        w = max(0, min(W-1, int(round(W*w_dec)-1)))
                        data[p, int(t/delta_t), h, w] = 1
            return data, label

        else:
            return data_numpy, label