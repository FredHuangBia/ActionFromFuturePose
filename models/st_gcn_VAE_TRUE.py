import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.utils.tgcn import ConvTemporalGraphical
from models.utils.graph import Graph


class Model_VAE(nn.Module):
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
                 edge_importance_weighting, pose=True, **kwargs):
        super().__init__()

        # build networks
        #self.data_bn = nn.BatchNorm1d_pose(in_channels * 25)
        #self.data_bn = nn.BatchNorm1d_velocities(in_channels * 25)
        
        # single LSTM
        self.pastencoder = nn.LSTM(25*3*2, 25*3, 1, batch_first = True)
        self.pastdecoder = nn.LSTM(25*3, 25*3, 1, batch_first = True)
        self.futureencoder = nn.LSTM(25*3, 25*3, 1, batch_first = True)
        self.futuredecoder = nn.LSTM(25*3*2, 25*3, 1, batch_first = True)
        
        # encode to mean and var
        self.latent_size = 75
        self.mean_fc = nn.Linear(25*3, self.latent_size)
        self.var_fc = nn.Linear(25*3, self.latent_size)

        # decoder
        self.decode_fc1 = nn.Linear(75, 75)
        self.relu_1 = nn.ReLU()
        self.decode_fc2 = nn.Linear(75, 75)
        

    def forward(self, pastpose, pastvelocities, futurepose, futurevelocities=None, T_future=None, train=True):
        
        # data normalization
        N, C, T_past, V, M = pastpose.size() # 1, 3, T, 25, M
        if T_future is None:
            T_future = futurepose.shape[2]
        #x = x.permute(0, 4, 3, 1, 2).contiguous()
        #x = x.view(N * M, V * C, T)
        #x = self.data_bn(x)
        #x = x.view(N, M, V, C, T) # 1, M, 25, 3, T
        #x = x.permute(0, 1, 3, 4, 2).contiguous()
        pastpose = pastpose.permute(0, 4, 1, 2, 3).contiguous()
        pastpose = pastpose.view(N * M, C, T_past, V) # BxM, 3, T, 25
        pastvelocities = pastvelocities.permute(0, 4, 1, 2, 3).contiguous()
        pastvelocities = pastvelocities.view(N * M, C, T_past, V) # BxM, 3, T, 25
        futurepose = futurepose.permute(0, 4, 1, 2, 3).contiguous()
        if train:
            futurepose = futurepose.view(N * M, C, T_future, V) # BxM, 3, T, 25
            futurevelocities = futurevelocities.permute(0, 4, 1, 2, 3).contiguous()
            futurevelocities = futurevelocities.view(N * M, C, T_future, V) # BxM, 3, T, 25
        else:
            futurepose = futurepose.view(N * M, C, 1, V) # BxM, 3, T, 25
        
        #get batch of frames
        pastpose = pastpose.permute(0, 2, 1, 3).contiguous()
        pastpose = pastpose.view(N * M, T_past, C * V) # BxM, T, 3*25
        pastvelocities = pastvelocities.permute(0, 2, 1, 3).contiguous()
        pastvelocities = pastvelocities.view(N * M, T_past, C * V) # BxM, T, 3*25
        futurepose = futurepose.permute(0, 2, 1, 3).contiguous()
        if train:
            futurepose = futurepose.view(N * M, T_future, C * V) # BxM, T, 3*25
            futurevelocities = futurevelocities.permute(0, 2, 1, 3).contiguous()
            futurevelocities = futurevelocities.view(N * M, T_future, C * V) # BxM, T, 3*25
        else:
            futurepose = futurepose.view(N * M, 1, C * V) # BxM, T, 3*25
        
        # forward

        #the past LSTM takes the past poses and velocities and encodes them into a hidden state
        _, (past_hidden,_) = self.pastencoder(torch.cat([pastpose, pastvelocities], dim=-1))
        past_hidden = past_hidden.view(1, N*M, 25*3)
        past_state = torch.zeros(past_hidden.shape).cuda()
        if train:
            pred_past, _ = self.pastdecoder(torch.flip(pastpose,dims=[1]), (past_hidden, past_state))
            pred_past = torch.flip(pred_past.view(N*M, T_past, 25*3),dims=[1])

            _, (future_enc, _) = self.futureencoder(futurevelocities, (past_hidden, past_state))
            future_enc = future_enc.view(1, N*M, 25*3)

            means = self.mean_fc(future_enc).view(1,N*M,25*3).permute(1,0,2)
            varis = self.var_fc(future_enc).view(1,N*M,25*3).permute(1,0,2)
            stds = torch.exp(0.5 * varis)
        
        # sample
        
        if train:
            samp = torch.randn([N*M, T_future, self.latent_size]).cuda()
            samp = samp * stds + means

            pred_future, _ = self.futuredecoder(torch.cat([futurepose,samp], dim=-1), (past_hidden, past_state))
            pred_future= pred_future.view(N*M, T_future, 25*3)

            past_predicted = self.relu_1(self.decode_fc1(pred_past))
            past_predicted = self.decode_fc2(past_predicted)        
            past_predicted = past_predicted.view(N, M, T_past, C, V).permute(0,3,2,4,1)

            future_predicted = self.relu_1(self.decode_fc1(pred_future))
            future_predicted = self.decode_fc2(future_predicted)        
            future_predicted = future_predicted.view(N, M, T_future, C, V).permute(0,3,2,4,1)

        else:
            #loop until all frames predicted
            cur_pose = futurepose
            pred_future = torch.zeros(N*M,T_future,C*V).cuda()
            for i in range(T_future):
                samp = torch.randn([N*M, 1, self.latent_size]).cuda()
                pred, _ = self.futuredecoder(torch.cat([cur_pose,samp], dim=-1), (past_hidden, past_state))
                pred_future[:,i,:] = pred.view(N*M, C*V)
                #these are velocities, get next pose
                cur_pose += pred_future[:,i,:].unsqueeze(1)

            future_predicted = self.relu_1(self.decode_fc1(pred_future))
            future_predicted = self.decode_fc2(future_predicted)        
            future_predicted = future_predicted.view(N, M, T_future, C, V).permute(0,3,2,4,1)
            
                
        
        if train:
            return past_predicted, future_predicted, means, varis
        else:
            return future_predicted