#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import model
import matplotlib.pyplot as plt
import dataset
import torch
from torch.utils.data import DataLoader

classes = ['person',
           'bird',
           'cat',
           'cow',
           'dog',
           'horse',
           'sheep',
           'aeroplane',
           'bicycle',
           'boat',
           'bus',
           'car',
           'motorbike',
           'train',
           'bottle',
           'chair',
           'diningtable',
           'pottedplant',
           'sofa',
           'tvmonitor']

dataset_path = './dataset'
use_fivecrop = True
model_path = './model/ResNet50_15005716.pt'
used_model = model.ResNet50


def demo():
    # load dataset and model

    device = torch.device('cuda')
    _, test_dataset = dataset.get_dataset(dataset_path, False, use_fivecrop)
    mymodel = used_model('none').to(device)
    mymodel.load(model_path)
    mymodel.eval()

    while True:
        index = int(input('enter index or exit(-1): '))
        if index == -1:
            break
        data = test_dataset[index]
        # process label
        image, label = data['image'].to(device), data['label'].to(device)
        if not use_fivecrop:
            image = image.unsqueeze(0)
        model_out = mymodel(image).mean(0)

        label = label.tolist()
        label_classes = []
        for i, v in enumerate(label):
            if v > 0:
                label_classes.append(classes[i])
        label_title = ' '.join(label_classes)

        # pred

        pred = model_out.tolist()
        pred_classes = []
        for i, v in enumerate(pred):
            if v > 0:
                pred_classes.append(classes[i])
        pred_title = ' '.join(pred_classes)

        # show result
        plt.imshow(test_dataset.get_raw_image(index))
        plt.title('pred: {} label: {}'.format(pred_title, label_title))
        plt.show()


if __name__ == "__main__":
    demo()
