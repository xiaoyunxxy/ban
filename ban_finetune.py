import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from collections import OrderedDict
from wanet_eval import eval as wa_eval
from iad_eval import eval as iad_eval
from bpp_eval import eval as bpp_eval
from ssdt_eval import eval as ssdt_eval

from train_models.IAD.networks.models import Generator
from train_models.ted.networks.models import Generator as Generator1
from train_models.IAD.dataloader import get_dataloader as iad_get_dataloader
from train_models.wanet.utils.dataloader import get_dataloader as wanet_get_dataloader

import models
import data.poison_cifar as poison
import data.poison_gtsrb as poison_gtsrb
import copy
from loader import dataset_loader

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg16'])
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--data-root", type=str, default="../data/")
parser.add_argument("--num-classes", type=int, default=10)
parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')

parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=200, help='the numbe of epoch for training')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')

# backdoor parameters
parser.add_argument('--clb-dir', type=str, default='', help='dir to training data under clean label attack')
parser.add_argument('--poison-type', type=str, default='badnets',
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=0.05,
                    help='proportion of poison examples in the training set')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')
parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--rob-lambda', type=float, default=0.2)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument("--attack_mode", type=str, default="all2one", help="all2one or all2all")
parser.add_argument("--device", type=str, default="cuda")


args = parser.parse_args()
args_dict = vars(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print(args)

    orig_train, clean_test = dataset_loader(args)

    # 0.1 for valid set, a part of valid set is used for perturbation and mitigation later
    sub_train, clean_train = random_split(dataset=orig_train, lengths=[0.05, 0.95], generator=torch.Generator().manual_seed(0))

    # parameters of dataset for subset
    clean_train.root = orig_train.root
    clean_train.targets = orig_train.targets
    if args.dataset == 'cifar10':
        clean_train.data = orig_train.data
    if args.dataset == 'gtsrb':
        clean_train.transform = orig_train.transform
        clean_train.target_transform =orig_train.target_transform
        clean_train.loader =orig_train.loader


    triggers = {'badnets': 'checkerboard_1corner',
                'clean-label': 'checkerboard_4corner',
                'blend': 'gaussian_noise',
                'benign': None}
    if args.poison_type in triggers.keys():
        trigger_type = triggers[args.poison_type]
        if args.poison_type in ['badnets', 'blend']:
            if args.dataset == 'gtsrb':
                poison_train, trigger_info = poison_gtsrb.add_trigger_gtsrb(data_set=clean_train, 
                    trigger_type=trigger_type, poison_rate=args.poison_rate,poison_target=args.poison_target)
                poison_test = poison_gtsrb.add_predefined_trigger_gtsrb(clean_test, trigger_info)
            elif args.dataset == 'cifar10':
                poison_train, trigger_info = \
                    poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
                                             poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
                poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
            else:
                raise ValueError('Check dataset.')
        elif args.poison_type == 'clean-label':
            poison_train = poison.CIFAR10CLB(root=args.clb_dir, transform=transform_train)
            pattern, mask = poison.generate_trigger(trigger_type=triggers['clean-label'])
            trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}
            poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
        elif args.poison_type == 'benign':
            poison_train = clean_train
            poison_test = clean_test
            trigger_info = None


        poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=8)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=8)
    clean_val_loader = DataLoader(sub_train, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=args.num_classes, norm_layer=models.NoisyBatchNorm2d).to(device)

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

    load_state_dict(net, state_dict)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    parameters = list(net.named_parameters())
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.eps / args.steps)

    if args.poison_type == 'iad':
        cl_test_acc, po_test_acc = iad_test(args, net, clean_test)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    elif args.poison_type == 'wanet':
        cl_test_acc, po_test_acc = wanet_test(args, net, clean_test)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    elif args.poison_type == 'bpp':
        cl_test_acc, po_test_acc = bpp_test(args, net, clean_test)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    elif args.poison_type == 'ssdt':
        cl_test_acc, po_test_acc = ssdt_test(args, net, clean_test)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    else:
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)

    print('Acc of the checkpoint (poisoned test set): {:.4}'.format(po_test_acc))
    print('Acc of the checkpoint (clean test set): {:.4}'.format(cl_test_acc))

    # Step 3: train backdoored models
    print('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')

    for epoch in range(1, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=clean_val_loader, noise_opt=noise_optimizer)

        if args.poison_type == 'iad':
            cl_test_acc, po_test_acc = iad_test(args, net, clean_test)
            cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss=0.0
        elif args.poison_type == 'wanet':
            cl_test_acc, po_test_acc = wanet_test(args, net, clean_test)
            cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss=0.0
        elif args.poison_type == 'bpp':
            cl_test_acc, po_test_acc = bpp_test(args, net, clean_test)
            cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss=0.0
        elif args.poison_type == 'ssdt':
            cl_test_acc, po_test_acc = ssdt_test(args, net, clean_test)
            cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss=0.0
        else:
            exclude_noise(net)
            cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)

        scheduler.step()
        end = time.time()
        print(
            '{:d} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))


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


def train(model, criterion, optimizer, data_loader, noise_opt):
    model.train()
    total_correct = 0
    total_loss = 0.0

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)


        # calculate the adversarial perturbation for neurons
        if args.eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)

                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()
                # clip_noise(model)


        output_noise = model(images)
        loss_rob = criterion(output_noise, labels)

        exclude_noise(model)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels) + args.rob_lambda*loss_rob

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def fea_mask_gen(model):
    if args.dataset == 'imagenet200':
        x = torch.rand(args.batch_size, 3, 224, 224).to(device)
    else:
        x = torch.rand(args.batch_size, 3, 32, 32).to(device)

    fea_shape = model.from_input_to_features(x, 0)
    rand_mask = torch.empty_like(fea_shape[0]).uniform_(0, 1).to(device)
    mask = torch.nn.Parameter(rand_mask.clone().detach().requires_grad_(True))
    return mask


def gene_mask(model, criterion, data_loader):
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
            loss = loss_positive - loss_negative + 0.25*mask_norm/mask_norm.item()

            
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

    return fea_mask.data


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
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


def wanet_test(opt, netC, data_set):
    opt.s=0.5
    opt.grid_rescale=1
    opt.target_label=0
    opt.cross_ratio=2
    opt.input_height = args.img_size
    opt.input_width = args.img_size
    opt.input_channel = args.channel

     # Dataset
    opt.bs = opt.batch_size
    # test_dl = wanet_get_dataloader(opt, False)

    test_set = copy.deepcopy(data_set)
    no_target_idx = (np.array([opt.target_label]) != test_set.targets)
    test_set.data = test_set.data[no_target_idx, :, :, :]
    test_set.targets = list(np.array(test_set.targets)[no_target_idx])
    test_dl = DataLoader(test_set, batch_size=opt.batch_size, num_workers=4)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, [30, 45], 0.1)

    # Load pretrained model
    # mode = opt.attack_mode
    # opt.ckpt_folder = os.path.join(opt.checkpoint, opt.dataset)
    # opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    # opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    
    if os.path.exists(opt.checkpoint):
        state_dict = torch.load(opt.checkpoint)
        # netC.load_state_dict(state_dict["netC"])
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
    else:
        print("Pretrained model doesnt exist")
        exit()

    
    acc_clean, acc_bd = wa_eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        noise_grid,
        identity_grid,
        opt,
    )
    return acc_clean, acc_bd


def iad_test(opt, netC, data_set):
    opt.lr_G = 1e-2
    opt.lr_C = 1e-2
    opt.lr_M =1e-2
    opt.schedulerG_milestones = [20, 30, 40, 50]
    opt.schedulerC_milestones = [30, 45]
    opt.schedulerM_milestones = [10, 20]
    opt.schedulerG_lambda = 0.1
    opt.schedulerC_lambda = 0.1
    opt.schedulerM_lambda = 0.1
    opt.n_iters = 60
    opt.lambda_div = 1
    opt.lambda_norm = 100

    opt.target_label = 0
    opt.p_attack = 0.1
    opt.p_cross = 0.1
    opt.mask_density = 0.032
    opt.EPSILON = 1e-7

    opt.random_rotation = 10
    opt.random_crop = 5

    opt.input_height = args.img_size
    opt.input_width = args.img_size
    opt.input_channel = args.channel
    opt.batchsize = opt.batch_size

    # path_model = os.path.join(
    #     opt.checkpoint, opt.dataset, opt.attack_mode, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset)
    # )
    state_dict = torch.load(opt.checkpoint)
    # print("load C")
    # netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()
    # netC.requires_grad_(False)
    # print("load G")
    netG = Generator(opt)
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)
    # print("load M")
    netM = Generator(opt, out_channels=1)
    netM.load_state_dict(state_dict["netM"])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)

    # Prepare dataloader
    # test_dl = iad_get_dataloader(opt, train=False)
    # test_dl2 = iad_get_dataloader(opt, train=False)

    test_set = copy.deepcopy(data_set)
    no_target_idx = (np.array([opt.target_label]) != test_set.targets)
    test_set.data = test_set.data[no_target_idx, :, :, :]
    test_set.targets = list(np.array(test_set.targets)[no_target_idx])
    test_dl = DataLoader(test_set, batch_size=opt.batch_size, num_workers=4)

    test_dl2 = copy.deepcopy(test_dl)

    acc_clean, acc_bd = iad_eval(netC, netG, netM, test_dl, test_dl2, opt)
    return acc_clean, acc_bd

def bpp_test(opt, model, data_set):
    test_set = copy.deepcopy(data_set)
    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    opt.dithering=False
    opt.squeeze_num=8
    opt.target_label=opt.poison_target

    no_target_idx = (np.array([opt.target_label]) != test_set.targets)
    test_set.data = test_set.data[no_target_idx, :, :, :]
    test_set.targets = list(np.array(test_set.targets)[no_target_idx])
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, num_workers=4)

    clean_acc, bd_acc = bpp_eval(
        model,
        test_loader,
        opt,
        )
    return clean_acc, bd_acc

def ssdt_test(opt, model, data_set):
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3

    netG = Generator1(opt).to(opt.device)

    netM = Generator1(opt, out_channels=1).to(opt.device)

    state_dict = torch.load(opt.checkpoint)

    netC=model
    netG.load_state_dict(state_dict["netG"])
    netM.load_state_dict(state_dict["netM"])

    # Prepare dataset
    # test_dl1 = get_dataloader(opt, train=False)
    # test_dl2 = get_dataloader(opt, train=False)
    opt.attack_mode='SSDT'
    opt.target_label=opt.poison_target
    opt.victim_label=1
    test_set = copy.deepcopy(data_set)

    # no_target_idx = (np.array([opt.target_label]) != test_set.targets)
    # test_set.data = test_set.data[no_target_idx, :, :, :]
    # test_set.targets = list(np.array(test_set.targets)[no_target_idx])
    test_dl1 = DataLoader(test_set, batch_size=opt.batch_size, num_workers=4)

    test_dl2 = copy.deepcopy(test_dl1)

    clean_acc, bd_acc = ssdt_eval(
            netC,
            netG,
            netM,
            test_dl1,
            test_dl2,
            opt
        )
    return clean_acc, bd_acc

if __name__ == '__main__':
    main()
