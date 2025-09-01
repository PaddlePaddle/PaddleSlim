# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

"""Augmentation
RandAug:
- reference: RandAugment: Practical automated data augmentation with a reduced search space
- https://arxiv.org/abs/1909.13719
AutoAug:
- reference: AutoAugment: Learning Augmentation Policies from Data
- https://arxiv.org/abs/1805.09501
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

LEVEL_DENOM = 10
#fill color is set to 128 instead fo image mean


def auto_augment_policy_v0():
    """policy v0:  hack from timm"""
    # ImageNet v0 policy from TPU EfficientNet impl, cannot find a paper reference.
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    policy = [[SubPolicy(*args) for args in subpolicy] for subpolicy in policy]
    return policy


def auto_augment_policy_v0r():
    """policy v0r:  hack from timm"""
    # ImageNet v0 policy from TPU EfficientNet impl, with variation of Posterize used
    # in Google research implementation (number of bits discarded increases with magnitude)
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('PosterizeIncreasing', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    policy = [[SubPolicy(*args) for args in subpolicy] for subpolicy in policy]
    return policy


def auto_augment_policy_originalr():
    """policy originalr:  hack from timm"""
    # ImageNet policy from https://arxiv.org/abs/1805.09501 with research posterize variation
    policy = [
        [('PosterizeIncreasing', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeIncreasing', 0.6, 7), ('PosterizeIncreasing', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeIncreasing', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    policy = [[SubPolicy(*args) for args in subpolicy] for subpolicy in policy]
    return policy


def auto_augment_policy_original():
    """25 types of augment policies in original paper"""
    policy = [
        [('PosterizeOriginal', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeOriginal', 0.6, 7), ('PosterizeOriginal', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeOriginal', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeOriginal', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    policy = [[SubPolicy(*args) for args in subpolicy] for subpolicy in policy]
    return policy


class AutoAugment():
    """Auto Augment
    Randomly choose a tuple of augment ops from a list of policy
    Then apply the tuple of augment ops to input image

    Examples:
        policy = auto_augment_policy_original()
        augment = AutoAugment(policy)
        transformed_image = augment(image)
    """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, image, policy_idx=None):
        if policy_idx is None:
            policy_idx = random.randint(0, len(self.policy) - 1)

        sub_policy = self.policy[policy_idx]
        for operation in sub_policy:
            image = operation(image)
        return image


def rand_augment_policy_increasing(prob=0.5, magnitude_idx=9, magnitude_std=0.5):
    """
    Rand augment policy: default rand-m9-mstd0.5-inc1
    """
    policy = [
        ('AutoContrast', prob, magnitude_idx, magnitude_std),
        ('Equalize', prob, magnitude_idx, magnitude_std),
        ('Invert', prob, magnitude_idx, magnitude_std),
        ('Rotate', prob, magnitude_idx, magnitude_std),

        ('PosterizeIncreasing', prob, magnitude_idx, magnitude_std),
        ('SolarizeIncreasing', prob, magnitude_idx, magnitude_std),
        ('SolarizeAdd', prob, magnitude_idx, magnitude_std),
        ('ColorIncreasing', prob, magnitude_idx, magnitude_std),
        ('ContrastIncreasing', prob, magnitude_idx, magnitude_std),
        ('BrightnessIncreasing', prob, magnitude_idx, magnitude_std),
        ('SharpnessIncreasing', prob, magnitude_idx, magnitude_std),

        ('ShearX', prob, magnitude_idx, magnitude_std),
        ('ShearY', prob, magnitude_idx, magnitude_std),
        ('TranslateX', prob, magnitude_idx, magnitude_std),
        ('TranslateY', prob, magnitude_idx, magnitude_std),
    ]
    policy = [SubPolicy(*args) for args in policy]
    return policy


class RandAugment():
    """Rand Augment
    Randomly choose N augment ops from a list of K policies
    Then apply the N ops to input image

    Examples:
        policy = rand_augment_policy_original(magnitude_idx)
        augment = RandAugment(policy)
        transformed_image = augment(image)
    """

    def __init__(self, policy, num_layers=2):
        """
        Args:
            policy: list of SubPolicy
            num_layers: int
        """
        self.policy = policy
        self.num_layers = num_layers

    def __call__(self, image):
        selected_idx = np.random.choice(len(self.policy), self.num_layers)

        for policy_idx in selected_idx:
            sub_policy = self.policy[policy_idx]
            image = sub_policy(image)
        return image


class SubPolicy:
    """Subpolicy
    Read augment name and magnitude, apply augment with probability
    Args:
        op_name: str, augment operation name
        prob: float, if prob > random prob, apply augment
        magnitude: int, index of magnitude in preset magnitude ranges
        magnitude_std: float, std of magnitude in preset magnitude ranges
    """

    def __init__(self, op_name, prob, magnitude, magnitude_std=0.5):
        image_ops = {
            'ShearX': shear_x,
            'ShearY': shear_y,
            'TranslateX': translate_x_absolute,
            'TranslateY': translate_y_absolute,
            'TranslateXRel': translate_x_relative,
            'TranslateYRel': translate_y_relative,
            'Rotate': rotate,
            'AutoContrast': auto_contrast,
            'Invert': invert,
            'Equalize': equalize,
            'Solarize': solarize,
            'SolarizeIncreasing': solarize,
            'SolarizeAdd': solarize_add,
            'Posterize': posterize,
            'PosterizeIncreasing': posterize,
            'PosterizeOriginal': posterize,
            'Contrast': contrast,
            'ContrastIncreasing': contrast,
            'Color': color,
            'ColorIncreasing': color,
            'Brightness': brightness,
            'BrightnessIncreasing': brightness,
            'Sharpness': sharpness,
            'SharpnessIncreasing': sharpness,
        }

        level_fn = {
            'ShearX': shear_level_to_arg,
            'ShearY': shear_level_to_arg,
            'TranslateX': translate_absolute_level_to_arg,
            'TranslateY': translate_absolute_level_to_arg,
            'TranslateXRel': translate_relative_level_to_arg,
            'TranslateYRel': translate_relative_level_to_arg,
            'Rotate': rotate_level_to_arg,
            'AutoContrast': None,
            'Invert': None,
            'Equalize': None,
            'Solarize': solarize_level_to_arg,
            'SolarizeIncreasing': solarize_increasing_level_to_arg,
            'SolarizeAdd': solarize_add_level_to_arg,
            'Posterize': posterize_level_to_arg,
            'PosterizeIncreasing': posterize_increasing_level_to_arg,
            'PosterizeOriginal': posterize_original_level_to_arg,
            'Contrast': enhance_level_to_arg,
            'ContrastIncreasing': enhance_increasing_level_to_arg,
            'Color': enhance_level_to_arg,
            'ColorIncreasing': enhance_increasing_level_to_arg,
            'Brightness': enhance_level_to_arg,
            'BrightnessIncreasing': enhance_increasing_level_to_arg,
            'Sharpness': enhance_level_to_arg,
            'SharpnessIncreasing': enhance_increasing_level_to_arg,
        }

        self.prob = prob
        self.magnitude = magnitude
        self.magnitude_std = magnitude_std

        self.ops = image_ops[op_name]
        self.level_fn = level_fn[op_name]

    def __call__(self, image):
        if self.prob < 1.0 and random.random() > self.prob:
            return image

        magnitude = self.magnitude
        # hack from timm auto_augment.py
        if self.magnitude_std > 0:
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        upper_bound = LEVEL_DENOM
        magnitude = max(0, min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude) if self.level_fn is not None else tuple()
        image = self.ops(image, *level_args)
        return image


#################################################################
# Convert level to Image op arguments
#################################################################
def randomly_negate(value):
    """negate the value with 0.5 prob"""
    return -value if random.random() > 0.5 else value


def shear_level_to_arg(level):
    # range [-0.3, 0.3]
    level = (level / LEVEL_DENOM) * 0.3
    level = randomly_negate(level)
    return (level,)


def translate_absolute_level_to_arg(level):
    # translate const = 100
    level = (level / LEVEL_DENOM) * 100.
    level = randomly_negate(level)
    return (level,)


def translate_relative_level_to_arg(level):
    # range [-0.45, 0.45]
    level = (level / LEVEL_DENOM) * 0.45
    level = randomly_negate(level)
    return (level,)


def rotate_level_to_arg(level):
    # range [-30, 30]
    level = (level / LEVEL_DENOM) * 30.
    level = randomly_negate(level)
    return (level,)


def solarize_level_to_arg(level):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return (int((level / LEVEL_DENOM) * 256),)


def solarize_increasing_level_to_arg(level):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return (256 - int((level / LEVEL_DENOM) * 256),)


def solarize_add_level_to_arg(level):
    # range [0, 110]
    return (int((level / LEVEL_DENOM) * 110),)


def posterize_level_to_arg(level):
    # range [0, 4]
    # intensity/severity of augmentation decreases with level
    return (int((level / LEVEL_DENOM) * 4),)


def posterize_increasing_level_to_arg(level):
    # range [4, 0]
    # intensity/severity of augmentation increases with level
    return (4 - int((level / LEVEL_DENOM) * 4),)


def posterize_original_level_to_arg(level):
    # range [4, 8]
    # intensity/severity of augmentation decreases with level
    return (int((level / LEVEL_DENOM) * 4) + 4,)


# For Contrast, Color, Brightness, Sharpness
def enhance_level_to_arg(level):
    # range [0.1, 1.9]
    return ((level / LEVEL_DENOM) * 1.8 + 0.1,)


# For ContrastIncreasing, ColorIncreasing, BrightnessIncreasing, SharpnessIncreasing
def enhance_increasing_level_to_arg(level):
    # range [0.1, 1.9]
    level = (level / LEVEL_DENOM) * 0.9
    level = max(0.1, 1.0 + randomly_negate(level))
    return (level,)


#################################################################
# PIL Image transforms
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform
#################################################################
def shear_x(image, factor, fillcolor=(128, 128, 128)):
    return image.transform(image.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), fillcolor=fillcolor)


def shear_y(image, factor, fillcolor=(128, 128, 128)):
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), fillcolor=fillcolor)


def translate_x_absolute(image, pixels, fillcolor=(128, 128, 128)):
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=fillcolor)


def translate_y_absolute(image, pixels, fillcolor=(128, 128, 128)):
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=fillcolor)


def translate_x_relative(image, pct, fillcolor=(128, 128, 128)):
    pixels = pct * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=fillcolor)


def translate_y_relative(image, pct, fillcolor=(128, 128, 128)):
    pixels = pct * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=fillcolor)


def rotate(image, degrees):
    return image.rotate(degrees)


def auto_contrast(image, magnitude=None):
    return ImageOps.autocontrast(image)


def invert(image, magnitude=None):
    return ImageOps.invert(image)


def equalize(image, magnitude=None):
    return ImageOps.equalize(image)


def solarize(image, thresh):
    return ImageOps.solarize(image, thresh)


def solarize_add(image, add, thresh=128):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)

    return image


def posterize(image, bits_to_keep):
    if bits_to_keep >= 8:
        return image
    return ImageOps.posterize(image, bits_to_keep)


def contrast(image, factor):
    return ImageEnhance.Contrast(image).enhance(factor)


def color(image, factor):
    return ImageEnhance.Color(image).enhance(factor)


def brightness(image, factor):
    return ImageEnhance.Brightness(image).enhance(factor)


def sharpness(image, factor):
    return ImageEnhance.Sharpness(image).enhance(factor)
