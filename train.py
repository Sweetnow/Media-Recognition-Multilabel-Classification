#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(model, op, loader, epoch, device, log_interval, pos_weight, save=True):
    model.train()
    start_time = time.time()
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    train_loss = 0
    batch_cnt = 0
    for batch_idx, data in enumerate(loader):
        image, label = data['image'].to(device), data['label'].to(device)
        op.zero_grad()
        output = loss(model(image), label)
        train_loss += output.item()
        batch_cnt += 1
        output.backward()
        op.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(label), len(loader.dataset),
                100. * batch_idx / len(loader), output.item()))
    print('Train Epoch: {}: time = {:d}s'.format(
        epoch, int(time.time()-start_time)))
    if save:
        model.save('{}.pt'.format(model.get_name()))
    return train_loss / batch_cnt
