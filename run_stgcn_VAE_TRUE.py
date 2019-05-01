import os
import sys
import yaml
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from models.st_gcn_VAE_TRUE import Model_VAE as STGCN
from data.ntu_fp import NTU_FP_Dataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config/fp_gcn/ntu_fp/train.yaml")
    argv = sys.argv[2:]
    p = parser.parse_args(argv)
    with open(p.config, 'r') as f:
        print("Loading config from:", p.config)
        args = yaml.load(f)

    train_feeder_args = args["train_feeder_args"]
    train_dataset = NTU_FP_Dataset(**train_feeder_args, debug=debug)
    train_loader = DataLoader(  dataset=train_dataset,
                                batch_size=args["batch_size"],
                                shuffle=True,
                                num_workers=2,
                                drop_last=True)

    test_feeder_args = args["test_feeder_args"]
    test_dataset = NTU_FP_Dataset(**test_feeder_args, debug=debug)   
    test_loader = DataLoader(  dataset=test_dataset,
                                batch_size=args["batch_size"],
                                shuffle=True,
                                num_workers=2,
                                drop_last=True) 

    devices = args["device"]
    model_args = args["model_args"]
    stgcn = STGCN(**model_args)
    stgcn.apply(weights_init)
    # model.load_state_dict(torch.load(save_path))
    if len(devices) > 1:
        stgcn = torch.nn.DataParallel(stgcn, device_ids=devices)
    stgcn = stgcn.cuda()

    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(
                stgcn.parameters(),
                lr=args["base_lr"],
                weight_decay=args["weight_decay"])

    work_dir = args["work_dir"]
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    train_log = os.path.join(work_dir, "train.log")
    test_log = os.path.join(work_dir, "test.log")
    train_logger = open(train_log, "w")
    test_logger = open(test_log, "w")

    num_future = 10
    
    num_epoch = args["num_epoch"]
    # if debug:
    #     num_epoch = 2

    train_len = len(train_loader)
    
    for epoch in range(num_epoch):
        # train
        stgcn.train()
        loss_value = []
        loss_mse_value = []
        for i, (data, label, length) in enumerate(train_loader):
            # get data
            data = data.float().cuda()
            label = label.long().cuda()
            # forward
            #print(data[:,:,-1:,:,:])
            past_pose_input = data[:,:, :-num_future, :, :]
            past_pose_output = data[:,:, 1:-num_future+1, :, :]
            past_velocities = data[:,:, 1:-num_future+1, :, :] - data[:,:, :-num_future, :, :]
            future_pose_input = data[:,:, -num_future:-1, :, :]
            future_pose_output = data[:,:, -num_future+1:, :, :]
            future_velocities = data[:,:, -num_future+1:, :, :] - data[:,:, -num_future:-1, :, :]
            
            #print(future_velocities.max(), future_velocities.min())
            
            past_predicted, future_predicted, means, varis = stgcn(past_pose_input, past_velocities, future_pose_input, future_velocities, train=True) 
            loss_mse_past = criterion_mse(past_predicted, past_velocities)
            loss_mse_future = criterion_mse(future_predicted, future_velocities)
            loss_mse = loss_mse_past+loss_mse_future
            loss = loss_mse*100 + (-0.5 * torch.sum(1+varis-means.pow(2)-varis.exp()))/10
            # backward
            # if (train_len*epoch+i) % args["loss_batch_size"]==0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                

            # statistics
            now_loss = loss.data.item() / (data.shape[2] - num_future)
            loss_value.append(now_loss)
            now_loss_mse = loss_mse.data.item() / (data.shape[2] - num_future)
            loss_mse_value.append(now_loss_mse)
            sys.stdout.write("Training epoch %d/%d batch %d/%d  batch loss: %.5f\r"
                            %(epoch+1, num_epoch, i, len(train_loader), now_loss))

        train_logger.write("epoch %d loss %f\n"%(epoch+1, np.mean(loss_value)))
        print("\nFinish training epoch %d/%d, epoch loss: %.8f, epoch mse loss: %.8f\n"
              %(epoch+1, num_epoch, np.mean(loss_value), np.mean(loss_mse_value)))

        # test
        stgcn.eval()
        loss_value = []
        loss_mse_value = []
        for i, (data, label, length) in enumerate(test_loader):
            # get data
            data = data.float().cuda()
            label = label.long().cuda()
            # forward
            past_pose_input = data[:,:, :-num_future, :, :]
            past_pose_output = data[:,:, 1:-num_future+1, :, :]
            past_velocities = data[:,:, 1:-num_future+1, :, :] - data[:,:, :-num_future, :, :]
            future_pose_input = data[:,:, -num_future, :, :].unsqueeze(2)
            future_pose_output = data[:,:, -num_future+1:, :, :]
            future_velocities = data[:,:, -num_future+1:, :, :] - data[:,:, -num_future:-1, :, :]
            future_predicted = stgcn(past_pose_input, past_velocities, future_pose_input, T_future = future_velocities.shape[2], train=False) 
            loss_mse = criterion_mse(future_predicted, future_velocities)
            loss = loss_mse*100

            # statistics
            now_loss = loss.data.item()  / (data.shape[2] - num_future)
            loss_value.append(now_loss)
            now_loss_mse = loss_mse.data.item() / (data.shape[2] - num_future)
            loss_mse_value.append(now_loss_mse)
            sys.stdout.write("Testing epoch %d/%d batch %d/%d  batch loss: %.5f \r"
                            %(epoch+1, num_epoch, i, len(test_loader), now_loss))
        test_logger.write("epoch %d loss %f\n"%(epoch+1, np.mean(loss_value)))
        print("\nFinish testing epoch %d/%d, epoch loss: %.8f, epoch mse loss: %.8f\n"
              %(epoch+1, num_epoch, np.mean(loss_value), np.mean(loss_mse_value)))

        # save trained model
        save_path = os.path.join(work_dir, 'trained_stgcn.pt')
        if isinstance(stgcn, nn.DataParallel):
            torch.save(stgcn.module.state_dict(), save_path)
        else:
            torch.save(stgcn.state_dict(), save_path)
