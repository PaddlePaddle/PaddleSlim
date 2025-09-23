# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset related classes and methods for ViT training and validation"""

import os
import math
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddle.vision import transforms
from paddle.vision import image_load
from .augment import auto_augment_policy_original
from .augment import AutoAugment
from .augment import rand_augment_policy_increasing
from .augment import RandAugment
from .random_erasing import RandomErasing


class ImageNet2012Dataset(Dataset):
    """Build ImageNet2012 dataset

    This class gets train/val imagenet datasets, which loads transfomed data and labels.
    Note:
        train_list.txt and val_list.txt are required.
        See the readme webpage for more info and download links: 
        https://github.com/BR-IDL/PaddleViT/tree/develop/image_classification#data-preparation

    Attributes:
        file_folder: path where imagenet images are stored
        transform: preprocessing ops to apply on image
        img_path_list: list of full path of images in whole dataset
        label_list: list of labels of whole dataset
    """

    def __init__(self, file_folder, is_train=True, transform_ops=None):
        """Init ImageNet2012 Dataset with dataset file path, mode(train/val), and transform"""
        super().__init__()
        self.file_folder = file_folder
        self.transforms = transform_ops
        self.img_path_list = []
        self.label_list = []

        list_name = 'train_list.txt' if is_train else 'val_list.txt'
        self.list_file = os.path.join(self.file_folder, list_name)
        assert os.path.isfile(self.list_file), f'{self.list_file} not exist!'

        with open(self.list_file, 'r') as infile:
            for line in infile:
                img_path = line.strip().split()[0]
                img_label = int(line.strip().split()[1])
                self.img_path_list.append(os.path.join(self.file_folder, img_path))
                self.label_list.append(img_label)
        print(f'----- Imagenet2012 {list_name} len = {len(self.label_list)}')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        data = image_load(self.img_path_list[index]).convert('RGB')
        data = self.transforms(data)
        label = self.label_list[index]

        return data, label

class ImageNet2012Dataset_(Dataset):
    """Build ImageNet2012 dataset

    This class gets train/val imagenet datasets, which loads transfomed data and labels.
    Note:
        train_list.txt and val_list.txt are required.
        See the readme webpage for more info and download links: 
        https://github.com/BR-IDL/PaddleViT/tree/develop/image_classification#data-preparation

    Attributes:
        file_folder: path where imagenet images are stored
        transform: preprocessing ops to apply on image
        img_path_list: list of full path of images in whole dataset
        label_list: list of labels of whole dataset
    """

    def __init__(self, file_folder,  img_path_list,label_list,transform_ops=None):
        """Init ImageNet2012 Dataset with dataset file path, mode(train/val), and transform"""
        super().__init__()
        self.file_folder = file_folder
        self.transforms = transform_ops
        self.img_path_list = img_path_list
        self.label_list = label_list


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        data = image_load(self.img_path_list[index]).convert('RGB')
        data = self.transforms(data)
        label = self.label_list[index]

        return data, label
def get_train_transforms_deit(config):
    """ Get full training transforms (as DeiT)
    For training, a RandomResizedCrop is applied with random mirror,
    then RandAug, AutoAug or ColorJitter is applied,
    then normalization is applied with mean and std,
    and RandomErase is applied.
    The input pixel values must be rescaled to [0, 1.].
    Outputs is converted to tensor.

    Args:
        config: configs contains IMAGE_SIZE, see config.py for details
    Returns:
        transforms_train: transform ops
    """
    aug_op_list = []
    # STEP1: random crop and resize
    aug_op_list.append(
        transforms.RandomResizedCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE),
                                     scale=(0.08, 1.0), interpolation='bicubic'))
    # STEP2: random horizontalflip
    aug_op_list.append(transforms.RandomHorizontalFlip())
    # STEP3: rand_augment or auto_augment or color jitter
    if config.TRAIN.RAND_AUGMENT: # MAE: True
        policy = rand_augment_policy_increasing(
            magnitude_idx=config.TRAIN.RAND_AUGMENT_MAGNITUDE)
        rand_augment = RandAugment(
            policy=policy, num_layers=config.TRAIN.RAND_AUGMENT_LAYERS)
        aug_op_list.append(rand_augment)
    elif config.TRAIN.AUTO_AUGMENT: # MAE: None
        policy = auto_augment_policy_original()
        auto_augment = AutoAugment(policy)
        aug_op_list.append(auto_augment)
    else: # MAE: None
        jitter = (float(config.TRAIN.COLOR_JITTER), ) * 3
        aug_op_list.append(transforms.ColorJitter(*jitter))
    # STEP3: other ops
    aug_op_list.append(transforms.ToTensor())
    aug_op_list.append(transforms.Normalize(mean=config.DATA.IMAGENET_MEAN,
                                            std=config.DATA.IMAGENET_STD))
    # STEP4: random erasing
    if config.TRAIN.RANDOM_ERASE_PROB > 0.:
        random_erasing = RandomErasing(prob=config.TRAIN.RANDOM_ERASE_PROB,
                                       mode=config.TRAIN.RANDOM_ERASE_MODE,
                                       max_count=config.TRAIN.RANDOM_ERASE_COUNT,
                                       num_splits=config.TRAIN.RANDOM_ERASE_SPLIT)
        aug_op_list.append(random_erasing)
    # Final: compose transforms and return
    transforms_train = transforms.Compose(aug_op_list)

    return transforms_train


def get_train_transforms_vit(config):
    """ Get training transforms
    For training, a RandomResizedCrop is applied with random mirror,
    then normalization is applied with mean and std.
    The input pixel values must be rescaled to [0, 1.].
    Outputs is converted to tensor.

    Args:
        config: configs contains IMAGE_SIZE, see config.py for details
    Returns:
        transforms_train: transform ops
    """
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(size=(config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE),
                                     interpolation='bicubic'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD)])
    return transforms_train


def get_val_transforms(config):
    """ Get training transforms
    For validation, image is first Resize then CenterCrop to image_size.
    Then normalization is applied with mean and std.
    The input pixel values must be rescaled to [0, 1.]
    Outputs is converted to tensor

    Args:
        config: configs contains IMAGE_SIZE, see config.py for details
    Returns:
        transforms_val: transform ops
    """
    scale_size = int(math.floor(config.DATA.IMAGE_SIZE / config.DATA.CROP_PCT))
    transforms_val = transforms.Compose([
        transforms.Resize(scale_size, 'bicubic'), # single int for resize shorter side of image
        transforms.CenterCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD)])
    return transforms_val


def get_dataset(config, is_train=True,model_type='vit'):
    """ Get dataset from config and mode (train/val)
    Returns the related dataset object according to configs and mode(train/val)

    Args:
        config: configs contains dataset related settings. see config.py for details
        is_train: bool, set True to use training set, otherwise val set. Default: True
    Returns:
        dataset: dataset object
    """
    if config.DATA.DATASET == "imagenet2012":
        if is_train:
            # use full augmentation for DeiT
            if model_type=='vit':
                transform_ops = get_train_transforms_vit(config)
            else:
                transform_ops = get_train_transforms_deit(config)
        else:
            transform_ops = get_val_transforms(config)
        dataset = ImageNet2012Dataset(config.DATA.DATA_PATH,
                                      is_train=is_train,
                                      transform_ops=transform_ops)
    else:
        raise NotImplementedError(
            "Wrong dataset name: [{config.DATA.DATASET}]. Only 'imagenet2012' is supported now")
    return dataset

def get_dataset_(config, img_path_list,label_list, is_train=True,model_type='vit'):
    """ Get dataset from config and mode (train/val)
    Returns the related dataset object according to configs and mode(train/val)

    Args:
        config: configs contains dataset related settings. see config.py for details
        is_train: bool, set True to use training set, otherwise val set. Default: True
    Returns:
        dataset: dataset object
    """
    if config.DATA.DATASET == "imagenet2012":
        if is_train:
            # use full augmentation for DeiT
            if model_type=='vit':
                transform_ops = get_train_transforms_vit(config)
            else:
                transform_ops = get_train_transforms_deit(config)
        else:
            transform_ops = get_val_transforms(config)
        dataset = ImageNet2012Dataset_(config.DATA.DATA_PATH, img_path_list,label_list,transform_ops=transform_ops)
    return dataset

def get_dataloader(config, dataset, is_train=True, use_dist_sampler=False):
    """Get dataloader from dataset, allows multiGPU settings.
    Multi-GPU loader is implements as distributedBatchSampler.

    Args:
        config: see config.py for details
        dataset: paddle.io.dataset object
        is_train: bool, when False, shuffle is off and BATCH_SIZE_EVAL is used, default: True
        use_dist_sampler: if True, DistributedBatchSampler is used, default: False
    Returns:
        dataloader: paddle.io.DataLoader object.
    """
    batch_size = config.DATA.BATCH_SIZE if is_train else config.DATA.BATCH_SIZE_EVAL

    if use_dist_sampler is True:
        sampler = DistributedBatchSampler(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=is_train,
                                          drop_last=is_train)
        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                num_workers=config.DATA.NUM_WORKERS)
    else:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=config.DATA.NUM_WORKERS,
                                shuffle=is_train,
                                drop_last=is_train)
    return dataloader

def get_dataloader_(config, dataset,batch_size, is_train=True, use_dist_sampler=False):

    if use_dist_sampler is True:
        sampler = DistributedBatchSampler(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=is_train,
                                          drop_last=is_train)
        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                num_workers=config.DATA.NUM_WORKERS)
    else:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=config.DATA.NUM_WORKERS,
                                shuffle=is_train,
                                drop_last=is_train)
    return dataloader
