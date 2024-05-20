import json
import os
import shutil
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from train_models.BppAttack.classifier_models import PreActResNet18, resnet18, DenseNet121, EfficientNetB0,MobileNetV2,vgg16,ResNeXt29_2x64d,SENet18
from train_models.BppAttack.networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from train_models.BppAttack.utils.dataloader import PostTensorTransform, get_dataloader
from train_models.BppAttack.utils.utils import progress_bar

import random
from numba import jit
from numba.types import float64, int64



def back_to_np(inputs,opt):
    
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb","celeba"]:
        expected_values = [0,0,0]
        variance = [1,1,1]
    inputs_clone = inputs.clone()
    print(inputs_clone.shape)
    if opt.dataset == "mnist":
        inputs_clone[:,:,:] = inputs_clone[:,:,:] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[channel,:,:] = inputs_clone[channel,:,:] * variance[channel] + expected_values[channel]
    return inputs_clone*255
    
def back_to_np_4d(inputs,opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb","celeba"]:
        expected_values = [0,0,0]
        variance = [1,1,1]
    inputs_clone = inputs.clone()
    
    if opt.dataset == "mnist":
        inputs_clone[:,:,:,:] = inputs_clone[:,:,:,:] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[:,channel,:,:] = inputs_clone[:,channel,:,:] * variance[channel] + expected_values[channel]

    return inputs_clone*255
    
def np_4d_to_tensor(inputs,opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb","celeba"]:
        expected_values = [0,0,0]
        variance = [1,1,1]
    inputs_clone = inputs.clone().div(255.0)

    if opt.dataset == "mnist":
        inputs_clone[:,:,:,:] = (inputs_clone[:,:,:,:] - expected_values[0]).div(variance[0])
    else:
        for channel in range(3):
            inputs_clone[:,channel,:,:] = (inputs_clone[:,channel,:,:] - expected_values[channel]).div(variance[channel])
    return inputs_clone
    
@jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


def eval(
    model,
    test_dl,
    opt,
):
    # print(" Eval:")
    squeeze_num = opt.squeeze_num
    
    model.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = model(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            inputs_bd = back_to_np_4d(inputs,opt)
            if opt.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i,:,:,:] = torch.round(torch.from_numpy(floydDitherspeed(inputs_bd[i].detach().cpu().numpy(),float(opt.squeeze_num))).cuda())

            else:
                inputs_bd = torch.round(inputs_bd/255.0*(squeeze_num-1))/(squeeze_num-1)*255

            inputs_bd = np_4d_to_tensor(inputs_bd,opt)
            
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
                
            # if batch_idx ==0:
                # print("backdoor target",targets_bd)
                # print("clean target",targets)

            preds_bd = model(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct / total_sample
            acc_bd = total_bd_correct / total_sample


    return acc_clean, acc_bd
