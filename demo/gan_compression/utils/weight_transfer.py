#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, Conv2DTranspose, InstanceNorm2D
from models.generator.modules import SeparableConv2D, MobileResnetBlock, ResnetBlock


### CoutCinKhKw
def transfer_Conv2D(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, Conv2D) and isinstance(m2, Conv2D)
    if m1.parameters()[0].shape[0] == 3:  ### last convolution
        assert input_index is not None
        m2.parameters()[0].set_value(m1.parameters()[0].numpy()[:,
                                                                input_index])
        if len(m2.parameters()) == 2:
            m2.parameters()[1].set_value(m1.parameters()[1].numpy())
        return None

    else:
        if m1.parameters()[0].shape[1] == 3:  ### first convolution
            assert input_index is None
            input_index = [0, 1, 2]
        p = m1.parameters()[0]

        if input_index is None:
            q = paddle.sum(paddle.abs(p), axis=[0, 2, 3])
            _, idx = paddle.topk(q, m2.parameters()[0].shape[1])
            p = p.numpy()[:, idx.numpy()]
        else:
            p = p.numpy()[:, input_index]

        if output_index is None:
            q = paddle.sum(paddle.abs(paddle.to_tensor(p)), axis=[1, 2, 3])
            _, idx = paddle.topk(q, m2.parameters()[0].shape[0])
            idx = idx.numpy()
        else:
            idx = output_index

        m2.parameters()[0].set_value(p[idx])
        if len(m2.parameters()) == 2:
            m2.parameters()[1].set_value(m1.parameters()[1].numpy()[idx])

        return idx


### CinCoutKhKw
def transfer_Conv2DTranspose(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, Conv2DTranspose) and isinstance(m2, Conv2DTranspose)
    assert output_index is None
    p = m1.parameters()[0]
    if input_index is None:
        q = paddle.sum(paddle.abs(p), axis=[1, 2, 3])
        _, idx = paddle.topk(q, m2.parameters()[0].shape[0])  ### Cin
        p = p.numpy()[idx.numpy()]
    else:
        p = p.numpy()[input_index]

    q = paddle.sum(paddle.abs(paddle.to_tensor(p)), axis=[0, 2, 3])
    _, idx = paddle.topk(q, m2.parameters()[0].shape[1])
    idx = idx.numpy()
    m2.parameters()[0].set_value(p[:, idx])
    if len(m2.parameters()) == 2:
        m2.parameters()[1].set_value(m1.parameters()[1].numpy()[idx])

    return idx


def transfer_SeparableConv2D(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, SeparableConv2D) and isinstance(m2, SeparableConv2D)
    dw1, pw1 = m1.conv[0], m1.conv[2]
    dw2, pw2 = m2.conv[0], m2.conv[2]

    if input_index is None:
        p = dw1.parameters()[0]
        q = paddle.sum(paddle.abs(p), axis=[1, 2, 3])
        _, idx = paddle.topk(q, dw2.parameters()[0].shape[0])
        input_index = idx.numpy()
    dw2.parameters()[0].set_value(dw1.parameters()[0].numpy()[input_index])

    if len(dw2.parameters()) == 2:
        dw2.parameters()[1].set_value(dw1.parameters()[1].numpy()[input_index])

    idx = transfer_Conv2D(pw1, pw2, input_index, output_index)
    return idx


def transfer_MobileResnetBlock(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, MobileResnetBlock) and isinstance(m2,
                                                            MobileResnetBlock)
    assert output_index is None
    idx = transfer_SeparableConv2D(
        m1.conv_block[1], m2.conv_block[1], input_index=input_index)
    idx = transfer_SeparableConv2D(
        m1.conv_block[6],
        m2.conv_block[6],
        input_index=idx,
        output_index=input_index)
    return idx


def transfer_ResnetBlock(m1, m2, input_index=None, output_index=None):
    assert isinstance(m1, ResnetBlock) and isinstance(m2, ResnetBlock)
    assert output_index is None
    idx = transfer_Conv2D(
        m1.conv_block[1], m2.conv_block[1], input_index=input_index)
    idx = transfer_Conv2D(
        m1.conv_block[6],
        m2.conv_block[6],
        input_index=idx,
        output_index=input_index)
    return idx


def transfer(m1, m2, input_index=None, output_index=None):
    assert type(m1) == type(m2)
    if isinstance(m1, Conv2D):
        return transfer_Conv2D(m1, m2, input_index, output_index)
    elif isinstance(m1, Conv2DTranspose):
        return transfer_Conv2DTranspose(m1, m2, input_index, output_index)
    elif isinstance(m1, ResnetBlock):
        return transfer_ResnnetBlock(m1, m2, input_index, output_index)
    elif isinstance(m1, MobileResnetBlock):
        return transfer_MobileResnetBlock(m1, m2, input_index, output_index)
    else:
        raise NotImplementedError('Unknown module [%s]!' % type(m1))


def load_pretrained_weight(model1, model2, netA, netB, ngf1, ngf2):
    assert model1 == model2
    assert ngf1 >= ngf2

    index = None
    if model1 == 'mobile_resnet_9blocks':
        assert len(netA.sublayers()) == len(netB.sublayers())
        for (n1, m1), (n2, m2) in zip(netA.named_sublayers(),
                                      netB.named_sublayers()):
            assert type(m1) == type(m2)
            if len(n1) > 8:
                continue
            if isinstance(m1, (Conv2D, Conv2DTranspose, MobileResnetBlock)):
                index = transfer(m1, m2, index)

    elif model1 == 'resnet_9blocks':
        assert len(netA.sublayers()) == len(netB.sublayers())
        for m1, m2 in zip(netA.sublayers(), netB.sublayers()):
            assert type(m1) == type(m2)
            if isinstance(m1, (Conv2D, Conv2DTranspose, ResnetBlock)):
                index = transfer(m1, m2, index)
    else:
        raise NotImplementedError('Unknown model [%s]!' % model1)
