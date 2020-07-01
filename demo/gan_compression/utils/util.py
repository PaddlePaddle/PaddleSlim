import os
import numpy as np
import pickle
from PIL import Image
import paddle.fluid as fluid


def load_network(model, model_path):
    if model_path.split('.')[-1] == 'pkl' or model_path.split('.')[
            -1] == 'pth':
        model_weight = pickle.load(open(model_path, 'rb'))
        for key, value in model_weight.items():
            model_weight[key] = np.array(value)
    else:
        assert os.path.exists(
            model_path + '.pdparams'), "model path: {} is not exist!!!".format(
                model_path + '.pdparams')
        model_weight, _ = fluid.load_dygraph(model_path)
    model.set_dict(model_weight)
    print("params {} load done".format(model_path))
    return model


def load_optimizer(optimizer, optimizer_path):
    assert os.path.exists(
        optimizer + '.pdopt'), "optimizer path: {} is not exist!!!".format(
            optimizer_path + '.pdopt')
    _, optimier_info = fluid.load_dygraph(optimizer_path)
    optimizer.set_dict(optimizer_info)
    return optimizer


def save_image(image, image_path):
    if len(image.shape) == 4:
        image = image[0]
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, 2)
    image_pil = Image.fromarray(image)
    image_pil.save(image_path)


def tensor2img(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2img(image_tensor[i], imtype, normalize))
        return image_numpy

    if len(image_tensor.shape) == 4:
        images_np = []
        for b in range(image_tensor.shape[0]):
            one_image = image_tensor[b]
            one_image_np = tensor2img(one_image)
            images_np.append(np.expand_dims(one_image_np, axis=0))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if len(image_tensor.shape) == 2:
        #image_tensor = fluid.layers.unsqueeze(image_tensor, axes=0)
        image_tensor = np.expand_dims(image_tensor, axis=0)
    if type(image_tensor) != np.ndarray:
        image_np = image_tensor.numpy()
    else:
        image_np = image_tensor
    if normalize:
        np.transpose(image_np, (1, 2, 0))
        image_np = (np.transpose(image_np, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_np = np.transpose(image_np, (1, 2, 0)) * 255.0
    image_np = np.clip(image_np, 0, 255)
    if image_np.shape[2] == 1:
        image_np = image_np[:, :, 0]
    return image_np.astype(imtype)
