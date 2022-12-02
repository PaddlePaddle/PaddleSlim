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
import sys
sys.path.append("../../")

import copy

import unittest
import paddle

# basic loss
from paddleslim.dygraph.dist.losses import CombinedLoss

# basic loss
from paddleslim.dygraph.dist.losses.basic_loss import DistanceLoss
from paddleslim.dygraph.dist.losses.basic_loss import CELoss
from paddleslim.dygraph.dist.losses.basic_loss import DMLLoss
from paddleslim.dygraph.dist.losses.basic_loss import RkdDistance
from paddleslim.dygraph.dist.losses.basic_loss import RKdAngle

# distillation loss
from paddleslim.dygraph.dist.losses import DistillationLoss

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

    def dist_np_distance_loss(self,
                              predicts,
                              loss_function=None,
                              mode="l2",
                              reduction="none",
                              model_name_pairs=(["", ""]),
                              key=None):
        loss_dict = dict()
        for idx, pair in enumerate(model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if key is not None:
                out1 = out1[key]
                out2 = out2[key]
            else:
                key = 0
            loss = self.np_distance_loss(
                out1, out2, mode=mode, reduction=reduction)
            loss_dict["{}_{}_{}_{}_{}".format(
                str(loss_function), pair[0], pair[1], key, key)] = loss

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
        x_feat_name = "student"
        y_feat_name = "teacher"
        pairs = [[x_feat_name, y_feat_name]]
        predicts = {
            "student": paddle.rand(shape),
            "teacher": paddle.rand(shape),
        }
        self.calc_distillation_distance_loss(predicts, pairs)

        predicts = {
            "student": {
                "feat": paddle.rand(shape),
            },
            "teacher": {
                "feat": paddle.rand(shape),
            },
        }
        self.calc_distillation_distance_loss(predicts, pairs, key="feat")

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
                    loss_func = DistillationLoss(
                        mode=mode,
                        loss_function='DistanceLoss',
                        model_name_pairs=pairs,
                        layers_name=[key, key] if key != None else None,
                        reduction=reduction)
                    np_result_dict = self.dist_np_distance_loss(
                        predicts,
                        loss_function='DistanceLoss',
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
            label = paddle.randint(0, class_num, shape=[batch_size, 1])

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


class TestDMLLoss(unittest.TestCase):
    """TestDMLLoss
    TestDMLLoss contains:
        1. unittest of basic loss
        2. unittest of distillation loss
    """

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

    def kldiv_loss(self, x, target, reduction="batchmean"):
        output = target * (np.log(target) - x)
        loss = np.where(target >= 0, output, np.zeros_like(x))

        if reduction == "batchmean":
            if len(x.shape) > 0:
                return loss.sum() / x.shape[0]
            else:
                return loss.sum()
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    def np_dml_loss(self, x, target, act="softmax"):
        if isinstance(x, paddle.Tensor):
            x = x.numpy()
        if isinstance(target, paddle.Tensor):
            target = target.numpy()
        soft_x = self.ref_softmax(x, axis=-1)
        soft_target = self.ref_softmax(target, axis=-1)

        log_soft_x = np.log(soft_x)
        log_soft_target = np.log(soft_target)
        loss = (self.kldiv_loss(log_soft_x, soft_target) + self.kldiv_loss(
            log_soft_target, soft_x)) / 2.0
        return loss

    def test_basic_dml_loss(self, ):
        batch_size = 32
        class_num = 1000

        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            x = paddle.rand([batch_size, class_num])
            target = paddle.rand([batch_size, class_num])

            loss_func = DMLLoss(act="softmax")
            pd_loss = loss_func(x, target).numpy()
            np_loss = self.np_dml_loss(x, target)
            self.assertTrue(np.allclose(np_loss, pd_loss))

    def dist_np_dml_loss(self,
                         predicts,
                         loss_function=None,
                         model_name_pairs=(["", ""]),
                         key=None):
        loss_dict = dict()
        for idx, pair in enumerate(model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if key is not None:
                out1 = out1[key]
                out2 = out2[key]
            else:
                key = 0
            loss_dict["{}_{}_{}_{}_{}".format(
                str(loss_function), pair[0], pair[1], key,
                key)] = self.np_dml_loss(out1, out2)
        return loss_dict

    def calc_distillation_dml_loss(self, predicts, pairs, key=None):
        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")

        for device in devices:
            paddle.set_device(device)
            loss_func = DistillationLoss(
                act="softmax",
                model_name_pairs=pairs,
                loss_function='DMLLoss',
                layers_name=[key, key] if key != None else None)
            np_result_dict = self.dist_np_dml_loss(
                predicts,
                model_name_pairs=pairs,
                loss_function='DMLLoss',
                key=key)
            pd_result_dict = loss_func(predicts, None)
            print(pd_result_dict.keys())
            print(np_result_dict.keys())
            for k in np_result_dict:
                pd_result = pd_result_dict[k].numpy()
                np_result = np_result_dict[k]
                self.assertTrue(np.allclose(np_result, pd_result))

    def test_distillation_dml_loss(self, ):
        shape = [20, 10]
        x_feat_name = "student"
        y_feat_name = "teacher"
        pairs = [[x_feat_name, y_feat_name]]
        predicts = {
            "student": paddle.rand(shape),
            "teacher": paddle.rand(shape),
        }
        self.calc_distillation_dml_loss(predicts, pairs, key=None)

        predicts = {
            "student": {
                "feat": paddle.rand(shape),
            },
            "teacher": {
                "feat": paddle.rand(shape),
            },
        }
        self.calc_distillation_dml_loss(predicts, pairs, key="feat")


class TestRKDLoss(unittest.TestCase):
    def pdist(self, e, squared=False, eps=1e-12):
        e_square = np.power(e, 2).sum(axis=1)
        prod = np.matmul(e, e.transpose())
        res = (
            np.expand_dims(e_square, 1) + np.expand_dims(e_square, 0) - 2 * prod
        ).clip(eps, sys.float_info.max)
        if not squared:
            res = np.sqrt(res)
        return res

    def p_normalize(self, x, axis=1, p=2, epsilon=1e-12, keepdims=True):
        xp = np.power(np.abs(x), p)
        s = np.sum(xp, axis=axis, keepdims=keepdims)
        r = np.maximum(np.power(s, 1.0 / p), epsilon)
        return x / r

    def np_smooth_l1_loss(self, x, y):
        diff = np.abs(x - y)
        diff_square = 0.5 * np.square(diff)
        loss = np.where(diff >= 1, diff - 0.5, diff_square).mean()
        return loss

    def np_rkd_distance(self, student, teacher, eps=1e-12):
        if isinstance(student, paddle.Tensor):
            student = student.numpy()
        if isinstance(teacher, paddle.Tensor):
            teacher = teacher.numpy()
        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        t_d = self.pdist(teacher, squared=False)
        mean_td = t_d.mean()
        t_d = t_d / (mean_td + eps)

        d = self.pdist(student, squared=False)
        mean_d = d.mean()
        d = d / (mean_d + eps)

        loss = self.np_smooth_l1_loss(d, t_d)
        return loss

    def np_rkd_angle(self, student, teacher):
        if isinstance(student, paddle.Tensor):
            student = student.numpy()
        if isinstance(teacher, paddle.Tensor):
            teacher = teacher.numpy()

        # reshape for feature map distillation
        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        td = np.expand_dims(teacher, 0) - np.expand_dims(teacher, 1)
        norm_td = self.p_normalize(td, axis=2, p=2)
        t_angle = np.matmul(norm_td, norm_td.transpose([0, 2, 1])).reshape(
            [-1, 1])

        sd = np.expand_dims(student, 0) - np.expand_dims(student, 1)
        norm_sd = self.p_normalize(sd, axis=2, p=2)
        s_angle = np.matmul(norm_sd, norm_sd.transpose([0, 2, 1])).reshape(
            [-1, 1])

        loss = self.np_smooth_l1_loss(s_angle, t_angle)
        return loss

    def test_rkd_distance_loss(self, ):
        batch_size = 32
        feat_dim = 100

        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            paddle.seed(0)
            x = paddle.rand([batch_size, feat_dim])
            y = paddle.rand([batch_size, feat_dim])

            loss_func = RkdDistance()
            pd_loss = loss_func(x, y).numpy()
            np_loss = self.np_rkd_distance(x, y)
            # NOTE: sqrt is included and seed is set for stability
            self.assertTrue(
                np.allclose(
                    np_loss, pd_loss, rtol=1e-5, atol=1e-07))

    def test_rkd_angle_loss(self, ):
        batch_size = 32
        feat_dim = 100

        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")
        for device in devices:
            paddle.set_device(device)
            paddle.seed(0)
            x = paddle.rand([batch_size, feat_dim])
            y = paddle.rand([batch_size, feat_dim])

            loss_func = RKdAngle()
            pd_loss = loss_func(x, y).numpy()
            np_loss = self.np_rkd_angle(x, y)
            # NOTE: sqrt is included and seed is set for stability
            self.assertTrue(np.allclose(np_loss, pd_loss))

    def dist_np_rkd_loss(
            self,
            predicts,
            model_name_pairs=(["", ""]),
            key=None,
            name="RKDLoss", ):
        loss_dict = dict()
        for idx, pair in enumerate(model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if key is not None:
                out1 = out1[key]
                out2 = out2[key]
            else:
                key = 0
            loss_dict["{}_{}_{}_{}_{}".format(name, pair[0], pair[
                1], key, key)] = self.np_rkd_angle(
                    out1, out2) + self.np_rkd_distance(out1, out2)

        return loss_dict

    def calc_distillation_rkd_loss(self, predicts, pairs, key=None):
        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")

        for device in devices:
            paddle.set_device(device)
            loss_func = DistillationLoss(
                model_name_pairs=pairs,
                loss_function='RKDLoss',
                layers_name=[key, key] if key != None else None)
            np_result_dict = self.dist_np_rkd_loss(
                predicts, model_name_pairs=pairs, key=key)
            pd_result_dict = loss_func(predicts, None)
            for k in np_result_dict:
                pd_result = pd_result_dict[k].numpy()
                np_result = np_result_dict[k]
                self.assertTrue(np.allclose(np_result, pd_result, rtol=1e-5))

    def test_distillation_rkd_loss(self, ):
        shape = [32, 16]
        x_feat_name = "student"
        y_feat_name = "teacher"
        pairs = [[x_feat_name, y_feat_name]]
        paddle.seed(0)
        predicts = {
            "student": paddle.rand(shape),
            "teacher": paddle.rand(shape),
        }
        self.calc_distillation_rkd_loss(predicts, pairs, key=None)

        predicts = {
            "student": {
                "feat": paddle.rand(shape),
            },
            "teacher": {
                "feat": paddle.rand(shape),
            },
        }
        self.calc_distillation_rkd_loss(predicts, pairs, key="feat")


class TestCombinedLoss(unittest.TestCase):
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

    def kldiv_loss(self, x, target, reduction="batchmean"):
        output = target * (np.log(target) - x)
        loss = np.where(target >= 0, output, np.zeros_like(x))

        if reduction == "batchmean":
            if len(x.shape) > 0:
                return loss.sum() / x.shape[0]
            else:
                return loss.sum()
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    def np_dml_loss(self, x, target, act="softmax"):
        if isinstance(x, paddle.Tensor):
            x = x.numpy()
        if isinstance(target, paddle.Tensor):
            target = target.numpy()
        soft_x = self.ref_softmax(x, axis=-1)
        soft_target = self.ref_softmax(target, axis=-1)

        log_soft_x = np.log(soft_x)
        log_soft_target = np.log(soft_target)
        loss = (self.kldiv_loss(log_soft_x, soft_target) + self.kldiv_loss(
            log_soft_target, soft_x)) / 2.0
        return loss

    def dist_np_dml_loss(self,
                         predicts,
                         model_name_pairs=(["", ""]),
                         loss_function=None,
                         key=None,
                         act="softmax"):
        loss_dict = dict()
        for idx, pair in enumerate(model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if key is not None:
                out1 = out1[key]
                out2 = out2[key]
            loss_dict["{}_{}_{}_{}_0".format(
                str(loss_function), pair[0], pair[1], idx)] = self.np_dml_loss(
                    out1, out2)
        return loss_dict

    def np_combined_loss(self, predicts, loss_cfg_list):
        # NOTE, dml is set as the list for combined loss
        loss_dict = dict()
        for idx, loss_func in enumerate(loss_cfg_list):
            cfg = copy.deepcopy(loss_func)
            weight = cfg.pop("weight")
            loss = self.dist_np_dml_loss(predicts, **cfg)

            if isinstance(loss, np.ndarray):
                loss = {
                    "{}_{}_{}".format(loss_func['loss_function'],
                                      str(loss), idx): loss
                }
            else:
                loss = {
                    "{}_{}".format(key, idx): loss[key] * weight
                    for key in loss
                }
            loss_dict.update(loss)
        loss_dict["loss"] = np.sum(list(loss_dict.values()))

        return loss_dict

    def test_combined_loss(self, ):
        shape = [32, 16]
        x_feat_name = "student"
        y_feat_name = "teacher"
        pairs = [[x_feat_name, y_feat_name]]
        paddle.seed(0)
        predicts = {
            "student": paddle.rand(shape),
            "teacher": paddle.rand(shape),
        }

        devices = ["cpu"]
        if paddle.is_compiled_with_cuda():
            devices.append("gpu")

        loss_cfg_list = [{
            "loss_function": "DMLLoss",
            "weight": 1.0,
            "act": "softmax",
            "model_name_pairs": pairs
        }, ]

        for device in devices:
            paddle.set_device(device)
            loss_func = CombinedLoss(loss_config_list=loss_cfg_list)
            pd_result_dict = loss_func(predicts, None)
            np_result_dict = self.np_combined_loss(predicts, loss_cfg_list)
            for k in pd_result_dict:
                pd_result = pd_result_dict[k].numpy()
                np_result = np_result_dict[k]
                self.assertTrue(np.allclose(np_result, pd_result))


if __name__ == '__main__':
    unittest.main()
