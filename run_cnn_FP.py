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

from models.cnn_FP import resnet50 as CNN
from data.ntu import NTU_Dataset

from utils.utils import acc_calculater


if __name__ == "__main__":
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config/cnn/ntu/train.yaml")
    argv = sys.argv[2:]
    p = parser.parse_args(argv)
    with open(p.config, 'r') as f:
        print("Loading config from:", p.config)
        args = yaml.load(f)

    train_feeder_args = args["train_feeder_args"]
    train_dataset = NTU_Dataset(**train_feeder_args, debug=debug)
    train_loader = DataLoader(  dataset=train_dataset,
                                batch_size=args["batch_size"],
                                shuffle=True,
                                num_workers=8,
                                drop_last=True)

    test_feeder_args = args["test_feeder_args"]
    test_dataset = NTU_Dataset(**test_feeder_args, debug=debug)   
    test_loader = DataLoader(  dataset=test_dataset,
                                batch_size=args["batch_size"],
                                shuffle=True,
                                num_workers=8,
                                drop_last=True) 

    devices = args["device"]
    model_args = args["model_args"]
    cnn = CNN(**model_args)

    if len(devices) > 1:
        cnn = torch.nn.DataParallel(cnn, device_ids=devices)
    cnn = cnn.cuda()

    criterion = nn.CrossEntropyLoss()
    params = cnn.parameters()
    # params = list(cnn.fc.parameters()) + list(cnn.att.parameters())
    optimizer = optim.Adam(
                params,
                lr=args["base_lr"],
                weight_decay=args["weight_decay"])

    work_dir = args["work_dir"]
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    train_log = os.path.join(work_dir, "train.log")
    test_log = os.path.join(work_dir, "test.log")
    train_logger = open(train_log, "w")
    test_logger = open(test_log, "w")

    acc_clc = acc_calculater([str(i) for i in range(60)])
    num_epoch = args["num_epoch"]

    for epoch in range(num_epoch):
        # train
        cnn.train()
        acc_clc.reset()
        loss_value = []
        for i, (data, label) in enumerate(train_loader):
            # get data
            data = data.float().cuda()
            label = label.long().cuda()
            # forward
            output = cnn(data) 
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                

            # statistics
            acc_clc.update(output.detach().cpu().numpy(), label.detach().cpu().numpy())
            now_loss = loss.data.item()
            loss_value.append(now_loss)
            sys.stdout.write("Training epoch %d/%d batch %d/%d  batch loss: %.5f\r"
                            %(epoch+1, num_epoch, i, len(train_loader), now_loss))

        train_logger.write("epoch %d loss %f\n"%(epoch+1, np.mean(loss_value)))
        print("\nFinish training epoch %d/%d, epoch loss: %.5f, epoch acc: %.5f\n"
              %(epoch+1, num_epoch, np.mean(loss_value), acc_clc.get_acc()))

        # save trained model
        save_path = os.path.join(work_dir, 'trained_cnn.pt')
        if isinstance(cnn, nn.DataParallel):
            torch.save(cnn.module.state_dict(), save_path)
        else:
            torch.save(cnn.state_dict(), save_path)

        # test
        cnn.eval()
        acc_clc.reset()
        loss_value = []
        for i, (data, label) in enumerate(test_loader):
            # get data
            data = data.float().cuda()
            label = label.long().cuda()
            # forward
            output = cnn(data)
            loss = criterion(output, label)
            # statistics
            acc_clc.update(output.detach().cpu().numpy(), label.detach().cpu().numpy())
            now_loss = loss.data.item()
            loss_value.append(now_loss)
            sys.stdout.write("Testing epoch %d/%d batch %d/%d  batch loss: %.5f \r"
                            %(epoch+1, num_epoch, i, len(test_loader), now_loss))
        test_logger.write("epoch %d loss %f \n"%(epoch+1, np.mean(loss_value)))
        print("\nFinish testing epoch %d/%d, epoch loss: %.5f, epoch acc: %.5f \n"
              %(epoch+1, num_epoch, np.mean(loss_value), acc_clc.get_acc()))
