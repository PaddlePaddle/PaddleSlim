# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import argparse
import paddle
from paddle.io import DataLoader
from paddle.metric import Accuracy
from data import ModelNetDataset
from model import PointNetClassifier


def parse_args():
    parser = argparse.ArgumentParser("Test")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--num_point", type=int, default=1024, help="point number")
    parser.add_argument(
        "--num_workers", type=int, default=32, help="num wrokers")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument(
        "--model_path", type=str, default="./BiPointNet.pdparams")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./modelnet40_normal_resampled", )
    parser.add_argument(
        "--binary",
        action='store_true',
        help="whehter to build binary pointnet")
    return parser.parse_args()


def test(args):

    test_data = ModelNetDataset(
        args.data_dir, split="test", num_point=args.num_point)
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size, )

    model = PointNetClassifier(binary=args.binary)
    if args.binary:
        import basic
        fp_layers = [
            id(model.feat.input_transfrom.conv1),
            id(model.feat.conv1),
            id(model.fc3)
        ]
        model = basic._to_bi_function(model, fp_layers=fp_layers)

        def func(model):
            if hasattr(model, "scale_weight_init"):
                model.scale_weight_init = True

        model.apply(func)

    model_state_dict = paddle.load(args.model_path)
    model.set_state_dict(model_state_dict)

    metrics = Accuracy()
    metrics.reset()
    model.eval()
    for iter, data in enumerate(test_loader):
        x, y = data
        pred, _, _ = model(x)

        correct = metrics.compute(pred, y)
        metrics.update(correct)
        if iter % args.log_freq == 0:
            print("Eval iter:", iter)
    test_acc = metrics.accumulate()
    print("Test Accuracy: {}".format(test_acc))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    test(args)
