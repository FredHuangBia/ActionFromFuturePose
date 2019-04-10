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

from models.st_gcn import Model as STGCN
from data.ntu import NTU_Dataset


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
    parser.add_argument('-c', '--config', default="config/st_gcn/ntu/train.yaml")
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
                                num_workers=2,
                                drop_last=True)

    test_feeder_args = args["test_feeder_args"]
    test_dataset = NTU_Dataset(**test_feeder_args, debug=debug)   
    test_loader = DataLoader(  dataset=train_dataset,
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

    criterion = nn.CrossEntropyLoss()
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

    num_epoch = args["num_epoch"]
    if debug:
        num_epoch = 2

    for epoch in range(num_epoch):
        # train
        loss_value = []
        correct = 0
        total = 0
        stgcn.train()
        for data, label in train_loader:
            # get data
            data = data.float().cuda()
            label = label.long().cuda()
            # forward
            output = stgcn(data)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            now_loss = loss.data.item()
            loss_value.append(now_loss)
            output = output.detach().cpu().numpy()
            pred = np.argmax(output, axis = 1)
            now_correct = np.sum(pred == label.cpu().numpy())
            now_acc = now_correct / len(label)
            total += len(label)
            correct += now_correct
            sys.stdout.write("Training epoch %d/%d   batch loss: %.5f   batch acc: %.5f \r"
                            %(epoch+1, num_epoch, now_loss, now_acc))
        train_logger.write("epoch %d loss %f acc %f\n"%(epoch+1, np.mean(loss_value), correct/total))
        print("\nFinish training epoch %d/%d, epoch loss: %.5f   epoch acc: %.5f \n"
              %(epoch+1, num_epoch, np.mean(loss_value), correct/total))

        # test
        loss_value = []
        correct = 0
        total = 0
        stgcn.eval()
        for data, label in test_loader:
            # get data
            data = data.float().cuda()
            label = label.long().cuda()
            # forward
            output = stgcn(data)
            loss = criterion(output, label)
            # statistics
            now_loss = loss.data.item()
            loss_value.append(now_loss)
            output = output.detach().cpu().numpy()
            pred = np.argmax(output, axis = 1)
            now_correct = np.sum(pred == label.cpu().numpy())
            now_acc = now_correct / len(label)
            total += len(label)
            correct += now_correct
            sys.stdout.write("Testing epoch %d/%d   batch loss: %.5f   batch acc: %.5f \r"
                            %(epoch+1, num_epoch, now_loss, now_acc))
        test_logger.write("epoch %d loss %f acc %f\n"%(epoch+1, np.mean(loss_value), correct/total))
        print("\nFinish testing epoch %d/%d, epoch loss: %.5f   epoch acc: %.5f \n"
              %(epoch+1, num_epoch, np.mean(loss_value), correct/total))

    # save trained moel
    save_path = os.path.join(work_dir, 'trained_stgcn.pt')
    if isinstance(stgcn, nn.DataParallel):
        torch.save(stgcn.module.state_dict(), save_path)
    else:
        torch.save(stgcn.state_dict(), save_path)
