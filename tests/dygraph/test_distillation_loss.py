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

# basic loss
from paddleslim.dygraph.dist.losses import DistanceLoss
from paddleslim.dygraph.dist.losses import CELoss

# distillation loss
from paddleslim.dygraph.dist.losses import DistillationDistanceLoss
from paddleslim.dygraph.dist.losses import DistillationRKDLoss
from paddleslim.dygraph.dist.losses import DistillationDMLLoss

import numpy as np


class TestDistanceLoss(unittest.TestCase):
    """TestDistanceLoss
    TestDistanceLoss contains:
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
        self.calc_distillation_distance_loss(predicts, pairs, key=None)

        predicts = {
            "feat_x": {
                "feat_loss": paddle.rand(shape),
            },
            "feat_y": {
                "feat_loss": paddle.rand(shape),
            },
        }
        self.calc_distillation_distance_loss(predicts, pairs, key="feat_loss")

    def calc_distillation_distance_loss(self, predicts, pairs, key=None):
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


class TestCELoss(unittest.TestCase):
    def stable_softmax(self, x):
        shiftx = (x - np.max(x)).clip(-64.)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def ref_softmax(self, x, axis=-1, dtype=None):
        if isinstance(x, paddle.Tensor):
            x = x.numpy()
        x_t = x.copy()
        if dtype is not None:
            x_t = x_t.astype(dtype)
        return np.apply_along_axis(self.stable_softmax, axis, x_t)

    def log_softmax(self, x, axis=-1):
        softmax_out = np.apply_along_axis(self.stable_softmax, axis, x)
        return np.log(softmax_out)

    def _cross_entropy_soft(self, softmax, label, axis, ignore_index=-1):
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    def np_cross_entropy_loss(self,
                              input,
                              label,
                              weight=None,
                              reduction='mean',
                              ignore_index=-100):
        log_softmax_out = self.log_softmax(input)
        input_shape = log_softmax_out.shape
        N = input_shape[0]
        out = np.zeros_like(label).astype(np.float64)
        total_weight = 0
        ###1. compute softmax cross_entropy (with weight)
        ###   Note: only support hard labels.
        for i in range(N):
            cur_target = label[i]
            if cur_target == ignore_index:
                out[i] = 0
                continue
            cur_weight = weight[cur_target] if weight is not None else 1
            total_weight += cur_weight
            out[i] = -log_softmax_out[i][cur_target] * cur_weight

        ###2. deal with reduction 
        if reduction == 'sum':
            return np.sum(out)
        elif reduction == 'mean':
            out = out.sum() / total_weight if total_weight != 0 else out.sum()
            return out
        elif reduction == 'none':
            return out

    def np_cross_entropy_soft(self,
                              x,
                              label,
                              axis=-1,
                              weight=None,
                              reduction='mean',
                              ignore_index=-100):
        if isinstance(x, paddle.Tensor):
            x = x.numpy()
        if isinstance(label, paddle.Tensor):
            label = label.numpy()
        softmax = self.ref_softmax(x, axis=axis)
        #1.loss
        loss = self._cross_entropy_soft(softmax, label, axis, ignore_index)

        if weight is None and reduction == 'none':
            return loss

        #2.weight
        weighted_loss = loss
        total_weight = softmax.shape[0]  # batch size
        if weight is not None:
            weighted_loss = np.zeros_like(loss).astype(np.float64)
            total_weight = 0
            for i in range(total_weight):
                cur_soft_label = label[i]
                cur_weight = np.dot(weight, cur_soft_label)
                total_weight += cur_weight
                weighted_loss[i] = loss[i] * cur_weight

        #3.reduce
        if reduction == 'none':
            return weighted_loss

        elif reduction == 'mean':
            weighted_loss_sum = np.sum(weighted_loss)
            weighted_loss_mean = weighted_loss_sum / total_weight
            return weighted_loss_mean

        else:
            weighted_loss_sum = np.sum(weighted_loss)
            return weighted_loss_sum

    def test_ce_loss_hard_label(self, ):
        batch_size = 16
        class_num = 1000

        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            x = paddle.rand([batch_size, class_num])
            label = paddle.randint(0, class_num, shape=[batch_size, ])

            loss_func = CELoss()
            pd_loss = loss_func(x, label).numpy()
            np_loss = self.np_cross_entropy_loss(x, label)
            self.assertTrue(np.allclose(np_loss, pd_loss))

    def test_ce_loss_soft_label(self, ):
        batch_size = 32
        class_num = 1000

        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            x = paddle.rand([batch_size, class_num])
            label = paddle.rand([batch_size, class_num])
            label = paddle.nn.functional.softmax(label, axis=-1)

            loss_func = CELoss(label_act=None)
            pd_loss = loss_func(x, label).numpy()
            np_loss = self.np_cross_entropy_soft(x, label)
            self.assertTrue(np.allclose(np_loss, pd_loss))


if __name__ == '__main__':
    unittest.main()
