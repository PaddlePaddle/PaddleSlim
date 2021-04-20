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

import json
import copy
import unittest
import numpy as np
from paddleslim.nas import GPNAS
from static_case import StaticCase

# 使用GP-NAS参加[CVPR 2021 NAS国际比赛](https://www.cvpr21-nas.com/competition) Track2 demo
# [CVPR 2021 NAS国际比赛Track2 studio地址](https://aistudio.baidu.com/aistudio/competition/detail/71?lang=en)
# [AI studio GP-NAS demo](https://aistudio.baidu.com/aistudio/projectdetail/1824958) 
# demo 基于paddleslim自研NAS算法GP-NAS:Gaussian Process based Neural Architecture Search 
# 基于本demo的改进版属于paddle解决方案，可以获得双倍奖金


class TestGPNAS(StaticCase):
    def test_gpnas(self):
        def preprare_trainning_data(file_name, t_flag):
            ## t_flag ==1 using all trainning data
            ## t_flag ==2 using half trainning data
            with open(file_name, 'r') as f:
                arch_dict = json.load(f)
            Y_all = []
            X_all = []
            for sub_dict in arch_dict.items():
                Y_all.append(sub_dict[1]['acc'] * 100)
                X_all.append(np.array(sub_dict[1]['arch']).T.reshape(4, 16)[2])
            X_all, Y_all = np.array(X_all), np.array(Y_all)
            X_train, Y_train, X_test, Y_test = X_all[0::t_flag], Y_all[
                0::t_flag], X_all[1::t_flag], Y_all[1::t_flag]
            return X_train, Y_train, X_test, Y_test

        stage1_file = './Track2_stage1_trainning.json'
        stage2_file = './Track2_stage2_few_show_trainning.json'
        X_train_stage1, Y_train_stage1, X_test_stage1, Y_test_stage1 = preprare_trainning_data(
            stage1_file, 1)
        X_train_stage2, Y_train_stage2, X_test_stage2, Y_test_stage2 = preprare_trainning_data(
            stage2_file, 2)
        gpnas = GPNAS(1, 1)
        w = gpnas.get_initial_mean(X_test_stage1, Y_test_stage1)
        init_cov = gpnas.get_initial_cov(X_train_stage1)
        error_list = np.array(
            Y_test_stage2.reshape(len(Y_test_stage2), 1) - gpnas.get_predict(
                X_test_stage2))
        print('RMSE trainning on stage1 testing on stage2:', np.sqrt(
            np.dot(error_list.T, error_list) / len(error_list)))
        gpnas.get_posterior_mean(X_train_stage2[0::3], Y_train_stage2[0::3])
        gpnas.get_posterior_mean(X_train_stage2[1::3], Y_train_stage2[1::3])
        gpnas.get_posterior_cov(X_train_stage2[1::3], Y_train_stage2[1::3])
        error_list = np.array(
            Y_test_stage2.reshape(len(Y_test_stage2), 1) -
            gpnas.get_predict_jiont(X_test_stage2, X_train_stage2[::1],
                                    Y_train_stage2[::1]))
        print('RMSE using stage1 as prior:', np.sqrt(
            np.dot(error_list.T, error_list) / len(error_list)))
        gpnas = GPNAS(2, 2)
        w = gpnas.get_initial_mean(X_test_stage1, Y_test_stage1)
        init_cov = gpnas.get_initial_cov(X_train_stage1)
        error_list = np.array(
            Y_test_stage2.reshape(len(Y_test_stage2), 1) - gpnas.get_predict(
                X_test_stage2))
        print('RMSE trainning on stage1 testing on stage2:', np.sqrt(
            np.dot(error_list.T, error_list) / len(error_list)))
        gpnas.get_posterior_mean(X_train_stage2[0::3], Y_train_stage2[0::3])
        gpnas.get_posterior_mean(X_train_stage2[1::3], Y_train_stage2[1::3])
        gpnas.get_posterior_cov(X_train_stage2[1::3], Y_train_stage2[1::3])
        error_list = np.array(
            Y_test_stage2.reshape(len(Y_test_stage2), 1) -
            gpnas.get_predict_jiont(X_test_stage2, X_train_stage2[::1],
                                    Y_train_stage2[::1]))
        print('RMSE using stage1 as prior:', np.sqrt(
            np.dot(error_list.T, error_list) / len(error_list)))


if __name__ == '__main__':
    unittest.main()
