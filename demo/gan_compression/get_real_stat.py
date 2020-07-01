import numpy as np
import argparse

from metric.inception import InceptionV3
from utils import util


def read_data(dataroot, filename):
    lines = open(os.path.join(dataroot, filename)).readlines()
    imgs = []
    for line in lines:
        img = Image.open(os.path.join(dataroot, line)).convert('RGB')
        img = (np.array(img).astype('float32') / 255.0 - 0.5) / 0.5
        img = img.transpose([2, 0, 1])
        imgs.append(img)


def main(cfgs):
    images = read_data(cfgs.dataroot, cfgs.filename)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])

    images = np.concatenate(images, axis=0)
    images = util.tensor2img(images).astype('float32')
    m2, s2 = _compute_statistic_of_img(images, inception_model, 32, 2048,
                                       cfgs.use_gpu, cfgs.inception_model_path)
    np.savez(cfgs.save_dir, mu=m2, sigma=s2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data',
        help="the dictionary of data")
    parser.add_argument(
        '--filename',
        type=str,
        default='trainA.txt',
        help="the name of list file")
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='Whether to use GPU')
    parser.add_argument(
        '--inception_model_path',
        type=str,
        default='metric/params_inceptionV3',
        help="The directory of inception pretrain  model")
    cfgs = parser.parse_args()

    main(cfgs)
