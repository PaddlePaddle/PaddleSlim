# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
sys.path.append("../")
import unittest
import numpy as np
import paddle
import paddle.nn as nn
from paddle.vision.models import mobilenet_v1
from paddleslim.nas.ofa.convert_super import Convert, supernet
from paddleslim.nas.ofa.utils import set_state_dict, dynabert_config
from paddleslim.nas.ofa.utils.nlp_utils import compute_neuron_head_importance, reorder_head, reorder_neuron
from paddleslim.nas.ofa import OFA


class TestModel(nn.Layer):
    def __init__(self):
        super(TestModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            312,
            12,
            1024,
            dropout=0.1,
            activation='gelu',
            attn_dropout=0.1,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, 3)
        self.fc = nn.Linear(312, 3)

    def forward(self, input_ids, segment_ids, attention_mask=[None, None]):
        src = input_ids + segment_ids
        out = self.encoder(src, attention_mask)
        out = self.fc(out[:, 0])
        return out


class TestComputeImportance(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.data_loader = self.init_data()

    def init_data(self):
        batch_size = 16
        hidden_size = 312
        d_model = 26
        input_ids = np.random.rand(batch_size, d_model,
                                   hidden_size).astype("float32")
        segment_ids = np.random.rand(batch_size, d_model,
                                     hidden_size).astype("float32")
        labels = np.random.randint(0, high=3, size=(batch_size, 1))
        data = ((paddle.to_tensor(input_ids), paddle.to_tensor(segment_ids),
                 paddle.to_tensor(labels)), )
        return data

    def reorder_neuron_head(self, model, head_importance, neuron_importance):
        # reorder heads and ffn neurons
        for layer, current_importance in enumerate(neuron_importance):
            # reorder heads
            idx = paddle.argsort(head_importance[layer], descending=True)
            reorder_head(model.encoder.layers[layer].self_attn, idx)
            # reorder neurons
            idx = paddle.argsort(
                paddle.to_tensor(current_importance), descending=True)
            reorder_neuron(model.encoder.layers[layer].linear1, idx, dim=1)
            reorder_neuron(model.encoder.layers[layer].linear2, idx, dim=0)

    def test_compute(self):
        head_importance, neuron_importance = compute_neuron_head_importance(
            task_name='xnli',
            model=self.model,
            data_loader=self.data_loader,
            num_layers=3,
            num_heads=12)
        assert (len(head_importance) == 3)
        assert (len(neuron_importance) == 3)
        self.reorder_neuron_head(self.model, head_importance, neuron_importance)


class TestComputeImportanceCase1(TestComputeImportance):
    def test_compute(self):
        for batch in self.data_loader:
            input_ids, segment_ids, labels = batch
            logits = self.model(
                input_ids, segment_ids, attention_mask=[None, None])
        assert logits.shape[1] == 3


class TestComputeImportanceCase2(TestComputeImportance):
    def test_compute(self):
        head_mask = paddle.ones(shape=[12], dtype='float32')
        for batch in self.data_loader:
            input_ids, segment_ids, labels = batch
            logits = self.model(
                input_ids, segment_ids, attention_mask=[None, head_mask])
        assert logits.shape[1] == 3


class TestSetStateDict(unittest.TestCase):
    def setUp(self):
        self.model = mobilenet_v1()
        self.origin_weights = {}
        for name, param in self.model.named_parameters():
            self.origin_weights[name] = param

    def test_set_state_dict(self):
        sp_net_config = supernet(expand_ratio=[0.5, 1.0])
        sp_model = Convert(sp_net_config).convert(self.model)
        set_state_dict(sp_model, self.origin_weights)


class TestSpecialConfig(unittest.TestCase):
    def test_dynabert(self):
        self.model = TestModel()
        sp_net_config = supernet(expand_ratio=[0.5, 1.0])
        self.model = Convert(sp_net_config).convert(self.model)
        ofa_model = OFA(self.model)
        config = dynabert_config(ofa_model, 0.5)


if __name__ == '__main__':
    unittest.main()
