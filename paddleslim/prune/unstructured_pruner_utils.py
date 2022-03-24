import numpy as np
import copy

__all__ = ["BLOCK_SPARSE_ACCURATE_THRESHOLD", "cal_mxn_avg_matrix"]

BLOCK_SPARSE_ACCURATE_THRESHOLD = 0.05


def cal_mxn_avg_matrix(mat, m=1, n=1):
    if m == 1 and n == 1: return copy.deepcopy(mat)
    assert m == n, "The block size m and n should be same, but got m={}, n={}".format(
        m, n)
    ori_row, ori_col = mat.shape[0], mat.shape[1]
    print(mat.shape)
    if len(mat.shape) == 4:
        assert mat.shape[2:] == (1, 1), "Only support for (n, n, 1, 1) for now."
        new_mat = mat.reshape(ori_row, ori_col)
    else:
        new_mat = np.array(mat)
    res_col = m - len(mat[0]) % m
    res_row = m - len(mat) % m

    new_mat = np.pad(new_mat, ((0, res_col), (0, res_col)), 'reflect')
    final_mat = np.zeros_like(new_mat)

    new_shape = [len(new_mat) // m, len(new_mat[0]) // m, m, m]
    strides = new_mat.itemsize * np.array(
        [len(new_mat) * m, m, len(new_mat), 1])
    new_mat = np.lib.stride_tricks.as_strided(
        new_mat, shape=new_shape, strides=strides)
    new_mat = new_mat.mean((2, 3), keepdims=True)
    new_mat = np.tile(new_mat, (1, 1, m, m))
    for i in range(len(new_mat)):
        sub_array = new_mat[i]
        tmp = np.concatenate(list(sub_array), axis=1)
        final_mat[i * m:i * m + m] = tmp
    final_mat = final_mat[:ori_row, :ori_col]
    if len(mat.shape) == 4:
        final_mat = final_mat.reshape(ori_row, ori_col, 1, 1)
    return final_mat
