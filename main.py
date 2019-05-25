#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import utils
import model
import dataset
import train
import test
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-label classification')
    parser.add_argument('--path', nargs='?', default='./dataset',
                        help='root directory')
    parser.add_argument('--model', nargs='?', default='ResNet50',
                        help='Choose Model: ResNet18, ResNet34, ResNet50, DenseNet')
    parser.add_argument('--num_fc_layers', nargs='?', type=int,
                        default=1, help='Model num_fc_layers')
    parser.add_argument('--frozen_layers', nargs='?', default='fc',
                        help='Model frozen_layers')
    parser.add_argument('--lr', nargs='?', type=int,
                        default=0.003, help='learning rate')
    parser.add_argument('--batch', nargs='?', type=int,
                        default=256, help='batch_size')
    parser.add_argument(
        '--augmentation', dest='use_augmentation', action='store_true')
    parser.add_argument('--no-augmentation',
                        dest='use_augmentation', action='store_false')
    parser.set_defaults(use_augmentation=False)
    parser.add_argument('--fivecrop', dest='use_fivecrop', action='store_true')
    parser.add_argument(
        '--no-fivecrop', dest='use_fivecrop', action='store_false')
    parser.set_defaults(use_fivecrop=True)
    parser.add_argument('--worker', nargs='?', type=int,
                        default=4, help='number of load workers')
    parser.add_argument('--early', nargs='?', type=int,
                        default=20, help='early stop')
    parser.add_argument('--epoch', nargs='?', type=int, default=100)
    parser.add_argument('--log', nargs='?', type=int, default=10)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.set_defaults(cuda=False)
    return parser.parse_args()


def main():
    args = parse_args()
    train_dataset, test_dataset = dataset.get_dataset(
        args.path, args.use_augmentation, args.use_fivecrop)
    train_loader = DataLoader(train_dataset, args.batch, True,
                              num_workers=args.worker, pin_memory=True)
    test_loader = DataLoader(train_dataset, args.batch, False,
                             num_workers=args.worker, pin_memory=True)
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.model == 'ResNet18':
        mymodel = model.ResNet18(
            args.num_fc_layers, args.frozen_layers).to(device)
    elif args.model == 'ResNet34':
        mymodel = model.ResNet34(
            args.num_fc_layers, args.frozen_layers).to(device)
    elif args.model == 'ResNet50':
        mymodel = model.ResNet50(
            args.num_fc_layers, args.frozen_layers).to(device)
    elif args.model == 'DenseNet':
        mymodel = model.DenseNet(args.num_fc_layers).to(device)
    else:
        pass
    op = optim.Adam(mymodel.parameters(), lr=args.lr)
    train_losses, test_mF1s, test_precisions, test_recalls = [], [], [], []
    early = args.early
    for i in range(args.epoch):
        train_loss = train.train(mymodel, op, train_loader, i,
                                 device, args.log, utils.pos_weight)
        mF1, recall, presicion = test.test(
            mymodel, test_loader, device, args.use_fivecrop)
        train_losses.append(train_loss)
        test_mF1s.append(mF1)
        test_precisions.append(presicion)
        test_recalls.append(recall)
        early = utils.early_stop(test_mF1s, early)
        if early <= 0:
            break
    utils.save_log(mymodel, train_losses, test_mF1s,
                   test_precisions, test_recalls)


if __name__ == '__main__':
    main()
