import numpy as np
from metric.compute_fid import _compute_statistic_of_img, _calculate_frechet_distance
from utils import util


def get_fid(fakes, model, npz, premodel_path, batch_size=1, use_gpu=True):
    m1, s1 = npz['mu'], npz['sigma']
    fakes = np.concatenate(fakes, axis=0)
    fakes = util.tensor2img(fakes).astype('float32')
    m2, s2 = _compute_statistic_of_img(fakes, model, batch_size, 2048, use_gpu,
                                       premodel_path)
    return float(_calculate_frechet_distance(m1, s1, m2, s2))
