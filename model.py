#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torchvision as tv


class Model(nn.Module):
    def __init__(self, name):
        self.name = name
        return super().__init__()

    def get_name(self):
        num_params = sum(param.numel()
                         for param in self.model.parameters() if param.requires_grad)
        return '{}_{}'.format(self.name, num_params)

    def load(self, backup):
        self.model.load_state_dict(torch.load(backup))

    def save(self, backup):
        torch.save(self.model.state_dict(), backup)

    def _freeze(self, layer_index):
        for i, child in enumerate(self.model.children()):
            if i < layer_index:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, images):
        x = images
        x = self.model(x)
        return x


class VGG16(Model):
    def __init__(self, num_fc_layers):
        super().__init__('VGG16')
        self.model = tv.models.vgg16(True)
        print(self.model)
        if num_fc_layers == 1:
            self.model.classifier = nn.Sequential(nn.Linear(512, 20))
        elif num_fc_layers == 2:
            self.model.classifier = nn.Sequential(nn.Linear(512, 200),
                                                  nn.ReLU(),
                                                  nn.Linear(200, 20))
        else:
            raise ValueError('num_fc_layers must be 1 or 2')
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._freeze(1)


class DenseNet(Model):
    def __init__(self):
        super().__init__('DenseNet')
        self.model = tv.models.densenet121(True)
        self.model.classifier = nn.Sequential(nn.Linear(1024, 20))
        self._freeze(1)

class ResNet18(Model):
    def __init__(self, frozen_layers):
        super().__init__('ResNet18')
        self.model = tv.models.resnet18(True)
        self.model.fc = nn.Sequential(nn.Linear(512, 20))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if frozen_layers == 'none':
            pass
        elif frozen_layers == 'fc':
            self._freeze(9)
        elif frozen_layers == 'one_conv':
            self._freeze(7)
        else:
            raise ValueError(
                'frozen_layers must be `none` `fc` and `one_conv`')

class ResNet34(Model):
    def __init__(self, frozen_layers):
        super().__init__('ResNet34')
        self.model = tv.models.resnet34(True)
        self.model.fc = nn.Sequential(nn.Linear(512, 20))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if frozen_layers == 'none':
            pass
        elif frozen_layers == 'fc':
            self._freeze(9)
        elif frozen_layers == 'one_conv':
            self._freeze(7)
        else:
            raise ValueError(
                'frozen_layers must be `none` `fc` and `one_conv`')

class ResNet50(Model):
    def __init__(self, frozen_layers):
        super().__init__('ResNet50')
        self.model = tv.models.resnet50(True)
        self.model.fc = nn.Sequential(nn.Linear(2048, 20))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if frozen_layers == 'none':
            pass
        elif frozen_layers == 'fc':
            self._freeze(9)
        elif frozen_layers == 'one_conv':
            self._freeze(7)
        else:
            raise ValueError(
                'frozen_layers must be `none` `fc` and `one_conv`')


if __name__ == '__main__':
    mymodel = VGG16(1)
    print(mymodel)
    print(mymodel.get_name())
