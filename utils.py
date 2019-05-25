#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import matplotlib.pyplot as plt
from json import dump, load

pos_weight = torch.tensor([2.0187, 15.0713,  8.8426, 31.0733,  7.1949, 25.2007, 27.5096, 16.1057,
                           19.1653, 23.6391, 21.8954,  9.1238, 21.8954, 17.6960, 14.1240,  7.5726,
                           15.5724, 19.2201, 14.5669, 18.9491], dtype=torch.float)


def show(train_losses, test_mF1s, test_precisions, test_recalls):
    x = range(len(train_losses))
    plt.subplot(2, 2, 1)
    plt.plot(x, train_losses, '.-')
    plt.title('train loss vs epochs')
    plt.ylabel('Train loss')
    plt.subplot(2, 2, 2)
    plt.plot(x, test_mF1s, '.-')
    plt.title('mF1-score vs epochs')
    plt.ylabel('mF1-score')
    plt.subplot(2, 2, 3)
    plt.plot(x, test_precisions, '.-')
    plt.title('test precision vs epochs')
    plt.ylabel('test precisions(%)')
    plt.subplot(2, 2, 4)
    plt.plot(x, test_recalls, '.-')
    plt.title('test recall vs epochs')
    plt.ylabel('test recalls(%)')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def early_stop(test_mF1s, early):
    if len(test_mF1s) >= 2 and test_mF1s[-1] < test_mF1s[-2]:
        return early-1
    else:
        return early


def save_log(model, train_losses, test_mF1s, test_precisions, test_recalls):
    log = {'train_losses': train_losses, 'test_mF1s': test_mF1s,
           'test_precisions': test_precisions, 'test_recalls': test_recalls}
    with open('{}.json'.format(model.get_name()), 'w') as f:
        dump(log, f)


def load_log(log_file):
    with open(log_file, 'r') as f:
        log = load(f)
    return list(log.values())


if __name__ == '__main__':
    import types

    class Test():
        def get_name(self):
            return 'model_test'

    model = Test()
    train_losses, test_mf1s, test_precisions, test_recalls = [
        1, 2, 3], [3, 2, 1], [4, 2, 1], [1, 3, 5]
    save_log(model, train_losses, test_mf1s, test_precisions, test_recalls)
    train_losses, test_mf1s, test_precisions, test_recalls = load_log(model)
    show(train_losses, test_mf1s, test_precisions, test_recalls)
