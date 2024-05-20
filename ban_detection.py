import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison_cifar
import data.poison_gtsrb as poison_gtsrb

from loader import dataset_loader, network_loader

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg16', 'DenseNet121'])
parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be perturbed.')
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--data-root", type=str, default="../data/")
parser.add_argument("--num-classes", type=int, default=10)
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--print-every', type=int, default=10, help='print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')

parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='benign',
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--acc-threshold', type=float, default=0.25)
parser.add_argument('--loss-threshold', type=float, default=4)
parser.add_argument('--mask-lambda', type=float, default=0.25)
parser.add_argument('--mask-lr', type=float, default=0.01)

parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--steps', type=int, default=1)


args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    if args.trigger_info:
        trigger_info = torch.load(args.trigger_info, map_location=device)
    else:
        if args.poison_type == 'benign':
            trigger_info = None
        else:
            triggers = {'badnets': 'checkerboard_1corner',
                        'clean-label': 'checkerboard_4corner',
                        'blend': 'gaussian_noise'}
            trigger_type = triggers[args.poison_type]
            pattern, mask = poison_cifar.generate_trigger(trigger_type=trigger_type)
            trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}


    orig_train, clean_test = dataset_loader(args)
    valid_frac = 0.1
    valia_set, _ = random_split(dataset=orig_train, lengths=[valid_frac, 1-valid_frac], generator=torch.Generator().manual_seed(0))
    sub_train, _ = random_split(dataset=orig_train, lengths=[args.val_frac, 1-args.val_frac], generator=torch.Generator().manual_seed(0))

    # print('number of samples in the valia: ', len(valia_set))
    print('number of samples in the sub_train: ', len(sub_train))

    random_sampler = RandomSampler(data_source=valia_set, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
    random_sampler_train = RandomSampler(data_source=sub_train, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
    sub_train_loader = DataLoader(sub_train, batch_size=args.batch_size, shuffle=False, sampler=random_sampler_train, num_workers=8)
    valid_loader = DataLoader(valia_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=8)
    
    try:
        state_dict = torch.load(args.checkpoint, map_location=device)['netC']
    except:
        try:
            state_dict = torch.load(args.checkpoint, map_location=device)['model_state_dict']
        except:
            try:
                state_dict = torch.load(args.checkpoint, map_location=device)['model']
            except:
                state_dict = torch.load(args.checkpoint, map_location=device)
    # state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(models, args.arch)(num_classes=args.num_classes, norm_layer=models.NoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader, args=args)
    print('Acc of the checkpoint (clean test set): {:.2f}'.format(cl_test_acc))

    non_perturb_acc = cl_test_acc

    parameters = list(net.named_parameters())
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.eps / args.steps)
    total_start = time.time()
    perturb_test_acc, l_pos, l_neg = perturbation_train(model=net, criterion=criterion, data_loader=sub_train_loader, noise_opt=noise_optimizer, clean_test_loader=valid_loader)
    total_end = time.time()
    print('total time: {:.4f}'.format(total_end-total_start))


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)



def clip_noise(model, lower=-args.eps, upper=args.eps):
    params = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d):
            module.reset(rand_init=rand_init, eps=args.eps)

def fea_mask_gen(model):
    if args.dataset == 'imagenet200':
        x = torch.rand(args.batch_size, 3, 224, 224).to(device)
    else:
        x = torch.rand(args.batch_size, 3, 32, 32).to(device)

    fea_shape = model.from_input_to_features(x, 0)
    rand_mask = torch.empty_like(fea_shape[0]).uniform_(0, 1).to(device)
    mask = torch.nn.Parameter(rand_mask.clone().detach().requires_grad_(True))
    return mask

def perturbation_train(model, criterion, noise_opt, data_loader, clean_test_loader):
    model.train()
    fea_mask = fea_mask_gen(model)
    opt_mask = torch.optim.Adam([fea_mask], lr=args.mask_lr)

    mepoch = 20

    for m in range(mepoch):
        start = time.time()
        total_mask_value = 0
        total_positive_loss = 0
        total_negative_loss = 0
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            opt_mask.zero_grad()

            features = model.from_input_to_features(images, 0)

            pred_positive = model.from_features_to_output(fea_mask*features, 0)
            pred_negative = model.from_features_to_output((1-fea_mask)*features, 0)
            mask_norm = torch.norm(fea_mask, 1)

            loss_positive = criterion(pred_positive, labels)
            loss_negative = criterion(pred_negative, labels) 
            loss = loss_positive - loss_negative + args.mask_lambda*mask_norm/mask_norm.item()

            
            total_mask_value += mask_norm.item()
            total_positive_loss += loss_positive.item()
            total_negative_loss += loss_negative.item()

            fea_mask.data = torch.clamp(fea_mask.data, min=0, max=1)

            loss.backward()
            opt_mask.step()

        l_pos = total_positive_loss/(batch_idx+1)
        l_neg = total_negative_loss/(batch_idx+1)
        end = time.time()
        print('mask epoch: {:d}'.format(m),
            '\tmask_norm: {:.4f}'.format(total_mask_value/(batch_idx+1)), 
            '\tloss_positive:  {:.4f}'.format(l_pos),
            '\tloss_negative:  {:.4f}'.format(l_neg),
            '\ttime:  {:.4f}'.format(end - start))

        
    print('\nGenerating noise perturbation.\n')
    start = time.time()
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # calculate the adversarial perturbation for neurons
        if args.eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.steps):
                noise_opt.zero_grad()

                include_noise(model)
                features_noise = model.from_input_to_features(images, 0)
                output_noise = model.from_features_to_output(features_noise, 0)

                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()
                # clip_noise(model)



    # cl_test_loss, cl_test_acc = mask_test(model=model, criterion=criterion, data_loader=clean_test_loader, args=args, mask=fea_mask.data)
    # print('Acc with mask (valid set): {:.4f}'.format(cl_test_acc))

    # exclude_noise(model)
    end = time.time()
    cl_test_loss, cl_test_acc = test(model=model, criterion=criterion, data_loader=clean_test_loader, args=args)
    print('Acc without mask (valid set): {:.4f}'.format(cl_test_acc))

    print('\n-------\n')
    # include_noise(model)
    cl_test_loss, cl_test_acc = mask_test(model=model, criterion=criterion, data_loader=clean_test_loader, args=args, mask=(1-fea_mask.data))
    print('Acc with negative mask (valid set): {:.4f}'.format(cl_test_acc))

    # cl_test_loss, cl_test_acc = mask_test(model=model, criterion=criterion, data_loader=clean_test_loader, args=args, mask=fea_mask.data)
    # print('Acc with positive mask (valid set): {:.4f}'.format(cl_test_acc))

    return cl_test_acc, l_pos, l_neg


def test(model, criterion, data_loader, args):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    reg = np.zeros([args.num_classes])
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

            for i in range(images.shape[0]):
                p = pred[i]
                reg[p] += 1

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    print('Prediction distribution: ', reg)
    print('Prediction targets to: ', np.argmax(reg))
    return loss, acc


def mask_test(model, criterion, data_loader, args, mask):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    reg = np.zeros([args.num_classes])
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            features = model.from_input_to_features(images, 0)
            output = model.from_features_to_output(mask*features, 0)

            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

            for i in range(images.shape[0]):
                p = pred[i]
                reg[p] += 1

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    print('Prediction distribution: ', reg)
    print('Prediction targets to: ', np.argmax(reg))
    return loss, acc


if __name__ == '__main__':
    main()
