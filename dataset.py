#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tfs
from PIL import Image


class PascalVOCDataset(Dataset):
    annotations_name = 'annotations.txt'
    images_dir = 'JPEGImages'
    train_prefix = ['2009', '2010', '2011']
    test_prefix = ['2007', '2008']
    suffix = '.jpg'
    CLASSES = 20

    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.images, self.labels = PascalVOCDataset.__load_annotation(
            os.path.join(root_dir, PascalVOCDataset.annotations_name), train)

    def __getitem__(self, index):
        image = self.get_raw_image(index)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': self.labels[index]}
        return sample

    def __len__(self):
        return len(self.images)

    def get_raw_image(self, index):
        image_name = os.path.join(
            self.root_dir, PascalVOCDataset.images_dir, self.images[index]+PascalVOCDataset.suffix)
        image = Image.open(image_name)
        return image

    def __load_annotation(path, train=True):
        '''
        load annotation 
        return image path array and ont-hot labels array
        '''
        images = []
        with open(path, 'r') as f:
            all_infos = list(map(lambda s: s[:-1].split(), f))
        if train:
            need_infos = list(
                filter(lambda l: l[0][:4] in PascalVOCDataset.train_prefix, all_infos))
        else:
            need_infos = list(
                filter(lambda l: l[0][:4] in PascalVOCDataset.test_prefix, all_infos))
        images = list(map(lambda l: l[0], need_infos))
        with torch.no_grad():
            indice_one_hot = torch.zeros(
                (len(images), PascalVOCDataset.CLASSES), dtype=torch.float32)
            for i in range(len(images)):
                indice = torch.LongTensor(
                    list(map(int, need_infos[i][1:])))
                indice_one_hot[i].scatter_(0, indice, 1.)
        return images, indice_one_hot


def get_dataset(path, use_augmentation, use_fivecrop):
    if use_augmentation:
        train_tfs = tfs.Compose([
            tfs.RandomCrop((300, 300), pad_if_needed=True),
            tfs.RandomHorizontalFlip(),
            tfs.RandomVerticalFlip(),
            tfs.RandomRotation(20),
            tfs.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            tfs.RandomGrayscale(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])
    else:
        train_tfs = tfs.Compose([
            tfs.Resize((300, 300)),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])
    if use_fivecrop:
        test_tfs = tfs.Compose([
            tfs.Resize((500, 500)),
            tfs.FiveCrop(300),
            tfs.Lambda(lambda crops: torch.stack([
                tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    tfs.ToTensor()(crop)) for crop in crops]))])
    else:
        test_tfs = tfs.Compose([
            tfs.Resize((300, 300)),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

    train_dataset = PascalVOCDataset(path, True, train_tfs)
    test_dataset = PascalVOCDataset(path, False, test_tfs)
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset('./dataset', True, False)
    print(len(train_dataset))
    print(test_dataset[0]['image'].shape)
