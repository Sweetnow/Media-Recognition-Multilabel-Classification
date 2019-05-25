#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
import dataset


def test(model, loader, device, use_fivecrop):
    model.eval()
    correct = torch.zeros(20, dtype=torch.float).to(device)
    label_cnt = torch.zeros(20, dtype=torch.float).to(device)
    predict_cnt = torch.zeros(20, dtype=torch.float).to(device)
    total_acc = torch.zeros(20, dtype=torch.float).to(device)
    weights = torch.zeros(20, dtype=torch.float).to(device)
    with torch.no_grad():
        for data in loader:
            image, label = data['image'].to(device), data['label'].to(device)
            if use_fivecrop:
                bs, ncrops, c, h, w = image.shape
                model_out = model(image.view(-1, c, h, w))
                model_out = model_out.view(bs, ncrops, -1).mean(1)
            else:
                model_out = model(image)
            pred = (model_out > 0).to(torch.float)

            total_acc += torch.sum(pred == label, 0).to(torch.float)
            weights += torch.sum(label, 0)
            correct += (pred*label).sum(0)
            label_cnt += label.sum(0)
            predict_cnt += pred.sum(0)

    weights /= torch.norm(weights, 1)
    class_acc = total_acc/len(loader.dataset)
    wacc = torch.sum(weights*class_acc).item()
    macc = torch.mean(class_acc).item()
    print('class_acc:', class_acc, ' wAcc:', wacc, 'mAcc:', macc)

    precision = correct/predict_cnt
    recall = correct/label_cnt
    f1 = 2*(precision*recall)/(precision+recall)
    wf1 = torch.sum(weights*f1).item()
    mf1 = torch.mean(f1).item()
    print('class f1', f1, ' wf1', wf1, ' mf1', mf1)

    test_precision = 100. * precision.mean().item()
    test_recall = 100. * recall.mean().item()
    print('\nTest set: recall: {}/{} ({:.0f}%), precision: {}/{} ({:.0f}%)\n'.format(
        correct.sum().item(), label_cnt.sum().item(),
        test_recall, correct.sum().item(), predict_cnt.sum().item(), test_precision))
    return mf1, test_recall, test_precision


if __name__ == "__main__":
    device = torch.device('cuda')
    mymodel = model.ResNet50('none').to(device)
    mymodel.load('./ResNet50_15005716.pt')
    _, test_dataset = dataset.get_dataset('./dataset', False, False)
    loader = DataLoader(test_dataset, 128, False,
                        num_workers=4, pin_memory=True)
    test(mymodel, loader, device, False)
