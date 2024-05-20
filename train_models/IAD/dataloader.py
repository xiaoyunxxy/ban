import csv
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)


class Smoothing(object):
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        im = np.asarray(img)
        im = cv2.GaussianBlur(im, (self.k, self.k), 0)
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(k={})".format(self.k)


def get_transform(opt, train=True, c=0, k=0):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
        if opt.dataset != "mnist":
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
        if opt.dataset == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if c > 0:
        transforms_list.append(ColorDepthShrinking(c))
    if k > 0:
        transforms_list.append(Smoothing(k))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb":
        pass
    elif opt.dataset == "celeba":
        mean = [0.50612009, 0.42543493, 0.38282761]
        std = [0.26589054, 0.24521921, 0.24127836]
        transforms_list.append(transforms.Normalize(mean, std))
    elif opt.dataset == "tiny-imagenet":
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        transforms_list.append(transforms.Normalize(mean, std))
    elif opt.dataset == "imagenet200":
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        transforms_list.append(transforms.Normalize(mean, std))
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


def get_dataloader(opt, train=True, c=0, k=0):
    transform = get_transform(opt, train, c=c, k=k)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "imagenet200":
        dataset = torchvision.datasets.ImageFolder(root=opt.data_root+'imagenet200/train' if train \
                                    else opt.data_root + 'imagenet200/val', transform=transform)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    elif opt.dataset == "tiny-imagenet":
        dataset = torchvision.datasets.ImageFolder(root=opt.data_root+'/tiny-imagenet-200/train' if train \
                                    else opt.data_root + '/tiny-imagenet-200/val', transform=transform)
    else:
        raise Exception("Invalid dataset")

    if train == True:
        sub_train, clean_train = random_split(dataset=dataset, lengths=[0.1, 0.9], generator=torch.Generator().manual_seed(0))
        dataset = clean_train

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True
    )
    return dataloader
