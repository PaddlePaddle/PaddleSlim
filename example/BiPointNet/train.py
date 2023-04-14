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

import os
import argparse
import paddle
from paddle.io import DataLoader
from paddle.metric import Accuracy
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay
from data import ModelNetDataset
from model import CrossEntropyMatrixRegularization, PointNetClassifier


def parse_args():
    parser = argparse.ArgumentParser("Train")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size in training")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate in training")
    parser.add_argument(
        "--num_point", type=int, default=1024, help="point number")
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="max epochs")
    parser.add_argument(
        "--num_workers", type=int, default=32, help="num wrokers")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument(
        "--pretrained",
        type=str,
        default='pointnet.pdparams',
        help='pretrained model path')
    parser.add_argument(
        "--save_dir", type=str, default='./save_model', help='save model path')
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./modelnet40_normal_resampled",
        help='dataset dir')
    parser.add_argument(
        "--binary",
        action='store_true',
        help="whehter to build binary pointnet")
    return parser.parse_args()


def train(args):
    train_data = ModelNetDataset(
        args.data_dir, split="train", num_point=args.num_point)
    test_data = ModelNetDataset(
        args.data_dir, split="test", num_point=args.num_point)
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size, )
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
        print(model)

    model_state_dict = paddle.load(path=args.pretrained)
    model.set_state_dict(model_state_dict)

    scheduler = CosineAnnealingDecay(
        learning_rate=args.learning_rate,
        T_max=args.max_epochs, )

    optimizer = Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay, )
    loss_fn = CrossEntropyMatrixRegularization()
    metrics = Accuracy()

    best_test_acc = 0
    for epoch in range(args.max_epochs):
        metrics.reset()
        model.train()
        for batch_id, data in enumerate(train_loader):

            x, y = data
            pred, trans_input, trans_feat = model(x)

            loss = loss_fn(pred, y, trans_feat)

            correct = metrics.compute(pred, y)
            metrics.update(correct)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if (batch_id + 1) % args.log_freq == 0:
                print("Epoch: {}, Batch ID: {}, Loss: {}, ACC: {}".format(
                    epoch, batch_id + 1, loss.item(), metrics.accumulate()))

        scheduler.step()

        metrics.reset()
        model.eval()
        for batch_id, data in enumerate(test_loader):
            x, y = data
            pred, trans_input, trans_feat = model(x)

            correct = metrics.compute(pred, y)
            metrics.update(correct)
        test_acc = metrics.accumulate()
        print("Test epoch: {}, acc is: {}".format(epoch, test_acc))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = os.path.join(args.save_dir, 'best_model.pdparams')
            paddle.save(model.state_dict(), save_path)
            print("Best Test ACC: {}, Model saved in {}".format(
                test_acc, save_path))
        else:
            print("Current Best Test ACC: {}".format(best_test_acc))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
