import os
import numpy as np
import torch
from copy import deepcopy
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import Compose
from torch.utils.data import Dataset
from PIL import Image
import PIL
import copy
import random
from torchvision.transforms import functional as F

class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        p_img= torch.clamp((1 - self.mask)*img + self.mask*self.pattern, min=0.0, max=255.0).type(torch.uint8)
        return p_img


class AddGTSRBTrigger(AddTrigger):
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        mask (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self, pattern, mask):
        super(AddGTSRBTrigger, self).__init__()

        if pattern is None:
            raise ValueError('pattern is None.')
        else:
            self.pattern = pattern

        if mask is None:
            raise ValueError('mask is None.')
        else:
            self.mask = mask

    def __call__(self, img):
        add_trigger = self.add_trigger

        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = add_trigger(img)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = add_trigger(img)
                img = img.numpy()
            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = add_trigger(img)
            # H x W x C
            else:
                img = img.permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


def split_dataset(dataset, val_frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split.
    :param val_frac: the fraction of validation set.
    :param perm: A predefined permutation for sampling. If perm is None, generate one.
    :return: A training set + a validation set
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_val = int(val_frac * len(dataset))

    # generate the training set
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_val:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_val:]].tolist()

    # generate the test set
    val_set = deepcopy(dataset)
    val_set.data = val_set.data[perm[:nb_val]]
    val_set.targets = np.array(val_set.targets)[perm[:nb_val]].tolist()
    return train_set, val_set


def generate_trigger(trigger_type):
    if trigger_type == 'checkerboard_1corner':  # checkerboard at the right bottom corner
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8) + 122
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for h in trigger_region:
            for w in trigger_region:
                pattern[30 + h, 30 + w, 0] = trigger_value[h+1][w+1]
                mask[30 + h, 30 + w, 0] = 1
    elif trigger_type == 'checkerboard_4corner':  # checkerboard at four corners
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for center in [1, 30]:
            for h in trigger_region:
                for w in trigger_region:
                    pattern[center + h, 30 + w, 0] = trigger_value[h + 1][w + 1]
                    pattern[center + h, 1 + w, 0] = trigger_value[h + 1][- w - 2]
                    mask[center + h, 30 + w, 0] = 1
                    mask[center + h, 1 + w, 0] = 1
    elif trigger_type == 'gaussian_noise':
        pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))
        mask = np.ones(shape=(32, 32, 1), dtype=np.uint8)
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask


def add_trigger_gtsrb(data_set, trigger_type, poison_rate, poison_target, trigger_alpha=1.0):
    """
    A simple implementation for backdoor attacks which only supports Badnets and Blend.
    :param clean_set: The original clean data.
    :param poison_type: Please choose on from [checkerboard_1corner | checkerboard_4corner | gaussian_noise].
    :param poison_rate: The injection rate of backdoor attacks.
    :param poison_target: The target label for backdoor attacks.
    :param trigger_alpha: The transparency of the backdoor trigger.
    :return: A poisoned dataset, and a dict that contains the trigger information.
    """
    pattern, mask = generate_trigger(trigger_type=trigger_type)

    p = torch.from_numpy(pattern)
    p = p.permute(2,0,1)[0]

    m = torch.from_numpy(mask)
    m = m.permute(2,0,1)[0]

    poison_set = PoisonedGTSRB(data_set, poison_target, poison_rate, p, m, 1.0, 2, 2)


    trigger_info = {'trigger_pattern': p, 'trigger_mask': m,
                    'trigger_alpha': trigger_alpha, 'poison_target': poison_target}
    return poison_set, trigger_info


def add_predefined_trigger_gtsrb(data_set, trigger_info, exclude_target=True):
    pattern = trigger_info['trigger_pattern']
    mask = trigger_info['trigger_mask']
    poison_target = trigger_info['poison_target']

    poison_rate = 1.0

    poison_set = PoisonedGTSRB(data_set, poison_target, poison_rate, pattern, mask, 1.0, 2, 2)

    return poison_set


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedGTSRB(ImageFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 mask,
                 alpha,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedGTSRB, self).__init__(
            benign_dataset.root,
            benign_dataset.transform,
            benign_dataset.target_transform,
            benign_dataset.loader)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddGTSRBTrigger(pattern, mask))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target





