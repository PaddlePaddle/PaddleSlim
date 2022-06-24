import os
import math
import random
import functools
import numpy as np
import paddle
from PIL import Image, ImageEnhance
import cv2
from paddle.io import Dataset

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 16
BUF_SIZE = 10240

DATA_DIR = 'data/ILSVRC2012/'
DATA_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], DATA_DIR)

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = pil_img_2_cv2(img)
    img = cv2.resize(
        img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    img = cv2_img_2_pil(img)
    return img


def pil_img_2_cv2(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def cv2_img_2_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def crop_image(img, target_size, center, central_fraction=0.875):
    width, height = img.size
    size = target_size
    if center == True:
        left = int((width - width * central_fraction) / 2.0)
        right = width - left
        top = int((height - height * central_fraction) / 2.0)
        bottom = height - top
        img = img.crop((left, top, right, bottom))
        img = pil_img_2_cv2(img)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2_img_2_pil(img)
    else:
        img = resize_short(img, target_size=256)
        width, height = img.size
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
        w_end = w_start + size
        h_end = h_start + size
        img = img.crop((w_start, h_start, w_end, h_end))
    return img


def rotate_image(img):
    angle = np.random.randint(-10, 11)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]
    try:
        img = Image.open(img_path)
    except:
        print(img_path, "not exists!")
        return None

    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = crop_image(img, target_size=DATA_DIM, center=False)
    else:
        img = crop_image(img, target_size=DATA_DIM, center=True)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.float32(img)
    img = img / 255.0

    img -= 0.5
    img *= 2.0

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR,
                    batch_size=1):
    def reader():
        try:
            with open(file_list) as flist:
                full_lines = [line.strip() for line in flist]
                if shuffle:
                    np.random.shuffle(full_lines)
                lines = full_lines
                for line in lines:
                    if mode == 'train' or mode == 'val':
                        img_path, label = line.split()
                        img_path = os.path.join(data_dir, img_path)
                        yield img_path, int(label) + 1
                    elif mode == 'test':
                        img_path = os.path.join(data_dir, line)
                        yield [img_path]
        except Exception as e:
            print("Reader failed!\n{}".format(str(e)))
            os._exit(1)

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def train(data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'train_list.txt')
    return _reader_creator(
        file_list,
        'train',
        shuffle=True,
        color_jitter=False,
        rotate=False,
        data_dir=data_dir)


def val(data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val_list.txt')
    return _reader_creator(file_list, 'val', shuffle=False, data_dir=data_dir)


def test(data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'test_list.txt')
    return _reader_creator(file_list, 'test', shuffle=False, data_dir=data_dir)
