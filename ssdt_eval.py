import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from train_models.ted.classifier_models import PreActResNet18, resnet18, PreActResNet34, VGG
from train_models.ted.networks.models import Generator, NetC_MNIST
from torch.utils.tensorboard import SummaryWriter
# from train_models.ted.utils import progress_bar
# import wandb


def create_bd(victim_inputs, victim_labels, netG, netM, opt):

    bd_labels = create_labels_bd(victim_labels, opt)
    patterns = create_patterns(netG, victim_inputs)
    masks_output = create_masks_output(netM, victim_inputs)
    bd_inputs = apply_masks_to_inputs(victim_inputs, patterns, masks_output)

    return bd_inputs, bd_labels, patterns, masks_output


def filter_victim_inputs_and_targets(inputs, labels, opt):

    victim_inputs = [input for input, label in zip(
        inputs, labels) if label == opt.victim_label]
    victim_labels = [label for label in labels if label == opt.victim_label]

    if not victim_inputs:
        return torch.empty(0, *inputs.shape[1:], device=inputs.device, dtype=torch.float), victim_labels

    return torch.stack(victim_inputs), torch.stack(victim_labels)


def filter_non_victim_inputs_and_targets(inputs, labels, opt):
    non_victim_inputs = [input for input, label in zip(
        inputs, labels) if label != opt.victim_label]
    non_victim_labels = [
        label for label in labels if label != opt.victim_label]
    if not non_victim_inputs:
        return torch.empty(0, *inputs.shape[1:], device=inputs.device, dtype=torch.float), non_victim_labels
    return torch.stack(non_victim_inputs), non_victim_labels


def create_labels_bd(victim_labels, opt):
    if opt.attack_mode == "SSDT":
        bd_targets = torch.tensor([opt.target_label for _ in victim_labels])
    else:
        raise Exception(
            "{} attack mode is not implemented".format(opt.attack_mode))

    return bd_targets.to(opt.device)


def create_patterns(netG, inputs):
    patterns = netG(inputs)
    return netG.normalize_pattern(patterns)


def create_masks_output(netM, inputs):
    return netM.threshold(netM(inputs))


def apply_masks_to_inputs(inputs, patterns, masks_output):
    return inputs + (patterns - inputs) * masks_output


def create_cross(inputs1, inputs2, netG, netM, opt):
    patterns2 = netG(inputs2)
    patterns2 = netG.normalize_pattern(patterns2)
    masks_output = netM.threshold(netM(inputs2))
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output


def eval(
        netC,
        netG,
        netM,
        test_dl1, 
        test_dl2,
        opt
):
    netC.eval()
    # print(" Eval:")
    total = 0.0

    total_correct_clean = 0.0
    total_correct_cross = 0.0

    total_victim = 0.0
    total_correct_bd = 0.0

    total_non_victim = 0.0
    total_correct_nvt = 0.0

    for batch_idx, (inputs1, labels1), (inputs2, labels2) in zip(range(len(test_dl1)), test_dl1, test_dl2):

        with torch.no_grad():
            inputs1, labels1 = inputs1.to(opt.device), labels1.to(opt.device)
            inputs2, labels2 = inputs2.to(opt.device), labels2.to(opt.device)
            bs = inputs1.shape[0]

            victim_inputs1, victim_labels1 = filter_victim_inputs_and_targets(
                inputs1, labels1, opt)

            if len(victim_inputs1) > 0:
                inputs_bd, labels_bd, patterns1, masks1 = create_bd(
                    victim_inputs1, victim_labels1, netG, netM, opt)

                total_victim += len(victim_labels1)
                preds_bd = netC(inputs_bd)
                preds_bd_label = torch.argmax(preds_bd, 1)
                correct_bd = torch.sum(preds_bd_label == labels_bd)
                total_correct_bd += correct_bd

            preds_clean = netC(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == labels1)
            total_correct_clean += correct_clean

            inputs_cross, _, _ = create_cross(
                inputs1, inputs2, netG, netM, opt)
            preds_cross = netC(inputs_cross)
            correct_cross = torch.sum(torch.argmax(preds_cross, 1) == labels1)
            total_correct_cross += correct_cross

            non_victim_inputs1, non_victim_labels1 = filter_non_victim_inputs_and_targets(
                inputs1, labels1, opt)
            if len(non_victim_labels1) > 0:
                total_non_victim += len(non_victim_labels1)
                inputs_nvt, targets_nvt, _, _ = create_bd(
                    non_victim_inputs1, non_victim_labels1, netG, netM, opt)
                preds_nvt = netC(inputs_nvt)
                preds_nvt_label = torch.argmax(preds_nvt, 1)
                correct_nvt = torch.sum(preds_nvt_label == torch.tensor(
                    non_victim_labels1).to(opt.device))
                total_correct_nvt += correct_nvt

            total += bs
            avg_acc_clean = total_correct_clean / total
            avg_acc_cross = total_correct_cross / total

            # print('----- total_victim. ', total_victim)
            # print('----- total_correct_bd. ', total_correct_bd)
            avg_acc_bd = total_correct_bd / total_victim


            avg_acc_nvt = total_correct_nvt / total_non_victim
            batch_acc_bd = correct_bd / len(victim_labels1)

            # infor_string = "Clean Acc: {:.3f} | BD Acc: {:.3f} | Cross Acc: {:.3f} | NVT Acc : {:.3f} | Batch BD Acc : {:.3f}".format(
            #     avg_acc_clean, avg_acc_bd, avg_acc_cross, avg_acc_nvt, batch_acc_bd
            # )
            # progress_bar(batch_idx, len(test_dl1), infor_string)

        # print("Clean Acc: {:.3f} | BD Acc: {:.3f} | Cross Acc: {:.3f} | NVT Acc : {:.3f} ".format(
        #     avg_acc_clean, avg_acc_bd, avg_acc_cross, avg_acc_nvt,
        # ))
        # wandb.log({
        #     "EvalCleanAcc": avg_acc_clean,
        #     "EvalBDAcc": avg_acc_bd,
        #     "EvalCrossAcc": avg_acc_cross,
        #     "EvalNVTAcc": avg_acc_nvt
        # }, step=epoch)


    return avg_acc_clean, avg_acc_bd