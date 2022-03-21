import numpy as np
import copy

__all__ = ["BLOCK_SPARSE_ACCURATE_THRESHOLD", "cal_mxn_avg_matrix"]

BLOCK_SPARSE_ACCURATE_THRESHOLD = 0.05


def cal_mxn_avg_matrix(mat, m=1, n=1):
    if m == 1 and n == 1: return copy.deepcopy(mat)
    avg_mat = np.zeros_like(mat)
    rows = len(mat) // m + 1
    cols = len(mat[0]) // n + 1
    for row in range(rows):
        for col in range(cols):
            avg_mat[m * row:m * row + m, n * col:n * col + n] = np.mean(mat[
                m * row:m * row + m, n * col:n * col + n])
    return avg_mat
