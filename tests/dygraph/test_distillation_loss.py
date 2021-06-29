# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("../../")

import unittest
import paddle

# basic loss
from paddleslim.dygraph.dist.losses import CombinedLoss
from paddleslim.dygraph.dist.losses import DistanceLoss

# distillation loss
from paddleslim.dygraph.dist.losses import DistillationDistanceLoss
from paddleslim.dygraph.dist.losses import DistillationRKDLoss
from paddleslim.dygraph.dist.losses import DistillationDMLLoss

import numpy as np


class TestDistanceLoss(unittest.TestCase):
    """
    loss test should contains:
        1. unittest of basic loss
        2. unittest of distillation loss
    """

    def np_distance_loss(self, x, y, mode="l2", reduction="none"):
        assert reduction in ["none", "mean", "sum"]
        if isinstance(x, paddle.Tensor):
            x = x.numpy()
        if isinstance(y, paddle.Tensor):
            y = y.numpy()
        if mode == "l2":
            diff = np.square(x - y)
        elif mode == "l1":
            diff = np.abs(x - y)
        elif mode == "smooth_l1":
            diff = np.abs(x - y)
            diff_square = 0.5 * np.square(diff)
            diff = np.where(diff >= 1, diff - 0.5, diff_square)

        if reduction == "none":
            out = diff
        elif reduction == "mean":
            out = np.mean(diff)
        elif reduction == "sum":
            out = np.sum(diff)
        return out

    def dist_np_distance_loss(
            self,
            predicts,
            mode="l2",
            reduction="none",
            model_name_pairs=(["", ""]),
            key=None,
            name="loss_distance", ):
        loss_dict = dict()
        for idx, pair in enumerate(model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if key is not None:
                out1 = out1[key]
                out2 = out2[key]
            loss = self.np_distance_loss(
                out1, out2, mode=mode, reduction=reduction)
            loss_dict["{}_{}_{}_{}_{}".format(name, mode, pair[0], pair[1],
                                              idx)] = loss

        return loss_dict

    def test_basic_distance_loss(self):
        shape = [10, 20]
        x = paddle.rand(shape)
        y = paddle.rand(shape)
        modes = ["l1", "l2", "smooth_l1"]
        reductions = ["none", "mean", "sum"]
        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            for reduction in reductions:
                for mode in modes:
                    np_result = self.np_distance_loss(
                        x, y, mode=mode, reduction=reduction)
                    loss_func = DistanceLoss(mode=mode, reduction=reduction)
                    pd_result = loss_func(x, y).numpy()
                    self.assertTrue(np.allclose(np_result, pd_result))

    def test_distillation_distance_loss(self, ):
        shape = [20, 10]
        x_feat_name = "feat_x"
        y_feat_name = "feat_y"
        pairs = [[x_feat_name, y_feat_name]]
        predicts = {
            "feat_x": paddle.rand(shape),
            "feat_y": paddle.rand(shape),
        }
        self.calc_dist_distance_loss(predicts, pairs, key=None)

        predicts = {
            "feat_x": {
                "feat_loss": paddle.rand(shape),
            },
            "feat_y": {
                "feat_loss": paddle.rand(shape),
            },
        }
        self.calc_dist_distance_loss(predicts, pairs, key="feat_loss")

    def calc_dist_distance_loss(self, predicts, pairs, key=None):
        modes = ["l1", "l2", "smooth_l1"]
        reductions = ["none", "mean", "sum"]
        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")

        for device in devices:
            paddle.set_device(device)
            for reduction in reductions:
                for mode in modes:
                    loss_func = DistillationDistanceLoss(
                        mode=mode,
                        model_name_pairs=pairs,
                        key=key,
                        reduction=reduction)
                    np_result_dict = self.dist_np_distance_loss(
                        predicts,
                        mode=mode,
                        reduction=reduction,
                        model_name_pairs=pairs,
                        key=key)
                    pd_result_dict = loss_func(predicts, None)
                    for k in np_result_dict:
                        pd_result = pd_result_dict[k].numpy()
                        np_result = np_result_dict[k]
                        self.assertTrue(np.allclose(np_result, pd_result))


if __name__ == '__main__':
    unittest.main()
