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
from paddleslim.dygraph.dist.losses import DistanceLoss, L2Loss

# distillation loss
from paddleslim.dygraph.dist.losses import DistillationDistanceLoss
from paddleslim.dygraph.dist.losses import DistillationRKDLoss
from paddleslim.dygraph.dist.losses import DistillationDMLLoss

import numpy as np


class TestL2Loss(unittest.TestCase):
    """
    loss test should contains:
        1. unittest of basic loss
        2. unittest of distillation loss
    """

    def np_l2_loss(self, x, y, reduction="none"):
        assert reduction in ["none", "mean", "sum"]
        if isinstance(x, paddle.Tensor):
            x = x.numpy()
        if isinstance(y, paddle.Tensor):
            y = y.numpy()
        diff = np.square(x - y)
        if reduction == "none":
            out = diff
        elif reduction == "mean":
            out = np.mean(diff)
        elif reduction == "sum":
            out = np.sum(diff)
        return out

    def dist_np_l2_loss(self,
                        predicts,
                        reduction="none",
                        model_name_pairs=(["", ""]),
                        key=None):
        for idx, pair in enumerate(model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[key]
                out2 = out2[key]
            loss = self.np_l2_loss(out1, out2, reduction=self.key)
            loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1],
                                           idx)] = loss

    def test_basic_l2_loss(self):
        shape = [10, 20]
        x = paddle.rand(shape)
        y = paddle.rand(shape)
        modes = ["none", "mean", "sum"]
        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            for mode in modes:
                np_result = self.np_l2_loss(x, y, reduction=mode)
                loss_func = L2Loss(reduction=mode)
                pd_result = loss_func(x, y).numpy()
                self.assertTrue(np.allclose(np_result, pd_result))

    def test_distillation_l2_loss_wo_key():
        shape = [20, 10]
        x_feat_name = "feat_x"
        y_feat_name = "feat_y"
        x_dict = {x_feat_name: paddle.rand(shape)}
        y_dict = {y_feat_name: paddle.rand(shape)}

        modes = ["none", "mean", "sum"]
        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            for mode in modes:
                np_result = self.np_l2_loss(x, y, reduction=mode)
                loss_func = L2Loss(reduction=mode)
                pd_result = loss_func(x, y).numpy()
                self.assertTrue(np.allclose(np_result, pd_result))

        loss_func = DistillationDistanceLoss(
            mode="l2", model_name_pairs=[["feat", "feat"]])

        shape = single


if __name__ == '__main__':
    unittest.main()
