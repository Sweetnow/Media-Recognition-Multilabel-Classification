#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import model
import matplotlib.pyplot as plt
import dataset
import torch

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


def demo():
    # load dataset and model
    use_fivecrop = False
    device = torch.device('cuda')
    _, test_dataset = dataset.get_dataset('./dataset', False, use_fivecrop)
    mymodel = model.ResNet50('none')
    mymodel.load('../model/ResNet50_15005716ww.pt')
    mymodel = mymodel.to(device)

    # index = 111    #person
    while(True):
        index=int(input('enter index:'))

        # process label
        model_input = test_dataset[index]
        image, label = model_input['image'].to(
            device), model_input['label'].to(device)
        label = (label > 0).to(torch.int).tolist()
        label_classes = []
        for i, v in enumerate(label):
            if v == 1:
                label_classes.append(classes[i])
        label_title = ' '.join(label_classes)

        # pred
        if not use_fivecrop:
            image=image.unsqueeze(0)
        pred = mymodel(image).mean(0)

        pred = (pred > 0).to(torch.int).tolist()
        pred_classes = []
        for i, v in enumerate(pred):
            if v == 1:
                pred_classes.append(classes[i])
        pred_title = ' '.join(pred_classes)

        # show result
        plt.imshow(test_dataset.get_raw_image(index))
        plt.title('pred: {} label: {}'.format(pred_title, label_title))
        plt.show()


if __name__ == "__main__":
    demo()
