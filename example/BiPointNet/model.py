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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

offset_map = {1024: -3.2041, 2048: -3.4025, 4096: -3.5836}


class TNet(nn.Layer):
    def __init__(self, k=64, binary=False):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1D(k, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.act_function = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k
        self.binary = binary
        self.iden = paddle.eye(self.k, self.k, dtype=paddle.float32)

    def forward(self, x):
        B, D, N = x.shape

        x = self.act_function(self.bn1(self.conv1(x)))
        x = self.act_function(self.bn2(self.conv2(x)))

        if self.binary:
            x = self.bn3(self.conv3(x))
            x = paddle.max(x, 2, keepdim=True) + offset_map[N]
        else:
            x = self.act_function(self.bn3(self.conv3(x)))
            x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = self.act_function(self.bn4(self.fc1(x)))
        x = self.act_function(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.reshape((-1, self.k, self.k)) + self.iden
        return x


class PointNetEncoder(nn.Layer):
    def __init__(self,
                 global_feat=True,
                 input_transform=True,
                 feature_transform=False,
                 channel=3,
                 binary=False):
        super(PointNetEncoder, self).__init__()

        self.global_feat = global_feat
        if input_transform:
            self.input_transfrom = TNet(k=channel)
        else:
            self.input_transfrom = lambda x: paddle.eye(
                channel, channel, dtype=paddle.float32
            )

        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)

        if feature_transform:
            self.feature_transform = TNet(k=64)
        else:
            self.feature_transform = lambda x: paddle.eye(64, 64, dtype=paddle.float32)

        self.act_function = nn.ReLU()
        self.binary = binary

    def forward(self, x):
        x = paddle.transpose(x, (0, 2, 1))
        B, D, N = x.shape
        trans_input = self.input_transfrom(x)
        x = paddle.transpose(x, (0, 2, 1))
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = paddle.bmm(x, trans_input)
        if D > 3:
            x = paddle.concat([x, feature], axis=2)
        x = paddle.transpose(x, (0, 2, 1))
        x = self.act_function(self.bn1(self.conv1(x)))

        trans_feat = self.feature_transform(x)
        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.bmm(x, trans_feat)
        x = paddle.transpose(x, (0, 2, 1))

        pointfeat = x
        x = self.act_function(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.binary:
            x = paddle.max(x, 2, keepdim=True) + offset_map[N]
        else:
            x = paddle.max(x, 2, keepdim=True)

        x = x.reshape((-1, 1024))

        if self.global_feat:
            return x, trans_input, trans_feat
        else:
            x = x.reshape((-1, 1024, 1)).repeat(1, 1, N)
            return paddle.cat([x, pointfeat], 1), trans_input, trans_feat


class PointNetClassifier(nn.Layer):
    def __init__(self, k=40, normal_channel=False, binary=False):
        super(PointNetClassifier, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(
            global_feat=True,
            input_transform=True,
            feature_transform=True,
            channel=channel,
            binary=binary, )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)
        self.act_function = nn.ReLU()
        self.binary = binary

    def forward(self, x):
        x, trans_input, trans_feat = self.feat(x)
        x = self.act_function(self.bn1(self.fc1(x)))
        x = self.act_function(self.bn2(self.fc2(x)))
        if not self.binary:
            x = self.dropout(x)
        x = self.fc3(x)
        return x, trans_input, trans_feat


class CrossEntropyMatrixRegularization(nn.Layer):
    def __init__(self, mat_diff_loss_scale=1e-3):
        super(CrossEntropyMatrixRegularization, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat=None):
        loss = F.cross_entropy(pred, target)

        if trans_feat is None:
            mat_diff_loss = 0
        else:
            mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

    def feature_transform_reguliarzer(self, trans):
        d = trans.shape[1]
        I = paddle.eye(d)
        loss = paddle.mean(
            paddle.norm(
                paddle.bmm(trans, paddle.transpose(trans, (0, 2, 1))) - I,
                axis=(1, 2)))
        return loss
