import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.utils.tgcn import ConvTemporalGraphical
from models.utils.graph import Graph

from models.st_gcn_FP import Model_FP

class Model_AR(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, pose=False, **kwargs):
        super().__init__()
        self.num_class = num_class
        self.model_fp = Model_FP(in_channels, num_class, graph_args, edge_importance_weighting, 
        							pose=pose, **kwargs)

        # self.att = nn.Linear(1600, 1)
        self.fc = nn.Linear(1600, num_class)
        self.pose = pose


    def forward(self, x):
        x = self.model_fp(x)
        if self.pose:
            return x
        seq_len = x.shape[1]
        x = x.view(-1, 1600)
        predicted = self.fc(x)
        # att = self.att(x)
        # att = F.softmax(att, dim=0)
        # predicted = predicted * att
        predicted = predicted.view(-1, seq_len, self.num_class)
        predicted = predicted.mean(dim=1)

        return predicted

    