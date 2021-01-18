# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import numpy as np
import copy

__all__ = ["GPNAS"]


class GPNAS(object):
    """
    GPNAS(Gaussian Process based Neural Architecture Search) is a neural architecture search algorithm. 
    We model the correlation between architectue and performance from a Bayesian perspective. Specifically, by introducing a novel Gaussian Process based
    NAS (GP-NAS) method, the correlations are modeled by the kernel function and mean function. The kernel function is also learnable to enable adaptive modeling for complex 
    correlations in different search spaces. Furthermore, by in-corporating a mutual information based sampling method, we can theoretically ensure the high-performance 
    architecture with only a small set of samples. After addressing these problems, training GP-NAS once enables direct performance prediction of any architecture in different 
    scenarios and may obtain efficient networks for different deployment platforms.
    """

    def __init__(self):

        self.hp_mat = 0.00001
        self.hp_cov = 0.1
        self.cov_w = None
        self.w = None
        self.c_flag = 1

    def _get_corelation(self, mat1, mat2):
        """
        give two typical kernel function
        
        Auto kernel hyperparameters estimation to be updated
        """

        mat_diff = abs(mat1 - mat2)

        if self.c_flag == 1:

            return 0.5 * np.exp(-np.dot(mat_diff, mat_diff) / 16)

        elif self.c_flag == 2:

            return 1 * np.exp(-np.sqrt(np.dot(mat_diff, mat_diff)) / 4)

    def _preprocess_X(self, X):
        """
        preprocess of input feature/ tokens of architecture
        more complicated preprocess can be added such as nonlineaer transformation
        """

        X = X.tolist()
        p_X = copy.deepcopy(X)

        for feature in p_X:
            feature.append(1)

        return p_X

    def _get_cor_mat(self, X):
        """
        get kernel matrix
        """
        X = np.array(X)
        l = X.shape[0]
        cor_mat = []

        for c_idx in range(l):
            col = []
            c_mat = X[c_idx].copy()

            for r_idx in range(l):
                r_mat = X[r_idx].copy()
                temp_cor = self._get_corelation(c_mat, r_mat)
                col.append(temp_cor)
            cor_mat.append(col)

        return np.mat(cor_mat)

    def get_predict(self, X):
        """
        get the prediction of network architecture X
        """

        X = self._preprocess_X(X)
        X = np.mat(X)

        return X * self.w

    def get_initial_mean(self, X, Y):
        """
        get initial mean of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        Y = np.mat(Y)
        self.w = np.linalg.inv(X.T * X + self.hp_mat * np.eye(X.shape[
            1])) * X.T * Y.T

        return self.w

    def get_initial_cov(self, X):
        """
        get initial coviarnce matrix of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        self.cov_w = self.hp_cov * np.eye(X.shape[1])

        return self.cov_w

    def get_posterior_mean(self, X, Y):
        """
        get posterior mean of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        Y = np.mat(Y)
        cov_mat = self._get_cor_mat(X)
        self.w = self.w - self.cov_w * X.T * np.linalg.inv(
            cov_mat + X * self.cov_w * X.T + self.hp_mat * np.eye(X.shape[
                0])) * (Y.T - X * w)

        return self.w

    def get_posterior_cov(self, X, Y):
        """
        get posterior coviarnce matrix of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        Y = np.mat(Y)
        cov_mat = self._get_cor_mat(X)
        self.cov_mat = np.linalg.inv(
            np.linalg.inv(X.T * cov_mat * X + self.hp_mat * np.eye(X.shape[1]))
            + np.linalg.inv(self.cov_w + self.hp_mat * np.eye(X.shape[
                1])) + self.hp_mat * np.eye(X.shape[1]))

        return self.cov_mat
