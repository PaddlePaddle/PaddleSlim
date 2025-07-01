import os
import paddle
import numbers
import paddle.vision.transforms as transforms
import paddle.vision.transforms.functional as F
from paddle.vision.datasets import Cifar10
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN
from paddle.io import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize([config.data.image_size]*2), transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize([config.data.image_size]*2),
                transforms.RandomHorizontalFlip(prob=0.5),
                transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0,
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize([config.data.image_size]*2), transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0]
        )

    # args.data_path = '/home/xingyu-zheng/laboratory/data/cifar10/cifar-10-python.tar.gz'
    args.data_path = "D:/Laboratory/data/cifar10/cifar-10-python.tar.gz"
    if config.data.dataset == "CIFAR10":
        dataset = Cifar10(
            # os.path.join(args.exp, "datasets", "cifar10"),
            data_file=args.data_path, 
            mode="train", 
            download=True,
            transform=tran_transform,
        )
        test_dataset = Cifar10(
            # os.path.join(args.exp, "datasets", "cifar10_test"),
            data_file=args.data_path, 
            mode="test", 
            download=True,
            transform=test_transform,
        )

    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize([config.data.image_size]*2),
                        transforms.RandomHorizontalFlip(),
                        transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0,
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize([config.data.image_size]*2),
                        transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0,
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize([config.data.image_size]*2),
                    transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0,
                ]
            ),
            download=True,
        )

    elif config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize([config.data.image_size]*2),
                        transforms.CenterCrop((config.data.image_size,)*2),
                        transforms.RandomHorizontalFlip(prob=0.5),
                        transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0,
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize([config.data.image_size]*2),
                        transforms.CenterCrop((config.data.image_size,)*2),
                        transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0,
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(args.exp, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize([config.data.image_size]*2),
                    transforms.CenterCrop((config.data.image_size,)*2),
                    transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0,
                ]
            ),
        )

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(prob=0.5), transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(transforms.Transpose(), lambda x: x if x.dtype != np.uint8 else x.astype('float32')/255.0),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return paddle.log(image) - paddle.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + paddle.rand(X.shape) / 256.0
    if config.data.gaussian_dequantization:
        X = X + paddle.randn(X.shape) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.unsqueeze(0)

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.unsqueeze(0)

    if config.data.logit_transform:
        X = paddle.nn.functional.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return paddle.clip(X, 0.0, 1.0)
