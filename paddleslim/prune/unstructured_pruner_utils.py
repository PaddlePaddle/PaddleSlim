import numpy as np
import copy

__all__ = ["BLOCK_SPARSE_ACCURATE_THRESHOLD", "cal_mxn_avg_matrix"]

BLOCK_SPARSE_ACCURATE_THRESHOLD = 0.05


def cal_mxn_avg_matrix(mat, m=1, n=1):
    if m == 1 and n == 1: return copy.deepcopy(mat)
    assert m == n, "The block size m and n should be same, but got m={}, n={}".format(
        m, n)
    ori_row, ori_col = mat.shape[0], mat.shape[1]
    if len(mat.shape) == 4:
        assert mat.shape[2:] == (1, 1), "Only support for (n, n, 1, 1) for now."
        mat = mat.reshape(ori_row, ori_col)

    res_col = m - len(mat[0]) % m
    res_row = m - len(mat) % m

    mat = np.pad(mat, ((0, res_col), (0, res_col)), 'reflect')
    avg_mat = np.zeros_like(mat)

    new_shape = [len(mat) // m, len(mat[0]) // m, m, m]
    strides = mat.itemsize * np.array([len(mat) * m, m, len(mat), 1])
    mat = np.lib.stride_tricks.as_strided(mat, shape=new_shape, strides=strides)
    mat = mat.mean((2, 3), keepdims=True)
    mat = np.tile(mat, (1, 1, m, m))
    for i in range(len(mat)):
        sub_array = mat[i]
        avg_mat[i * m:i * m + m] = np.concatenate(list(sub_array), axis=1)
    avg_mat = avg_mat[:ori_row, :ori_col]
    if len(mat.shape) == 4:
        avg_mat = avg_mat.reshape(ori_row, ori_col, 1, 1)
    return avg_mat
