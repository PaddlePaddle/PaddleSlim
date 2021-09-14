#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn

from .basic_loss import BASIC_LOSS

__all__ = ["DistillationLoss"]


class DistillationLoss(nn.Layer):
    """
    DistillationLoss
    Args:
        model_name_pairs(list | tuple): model name pairs to extract submodel output.
        layers_name(list(string)): keys of the tensor used to calculate loss if the submodel.
        loss_function(string): the name of loss function.
        temperature(float): the temperature to compute distill loss.
    """

    def __init__(self,
                 model_name_pairs=[],
                 layers_name=None,
                 loss_function=None,
                 temperature=1.0,
                 **params):
        super().__init__()
        self.model_name_pairs = model_name_pairs
        self.layers_name = layers_name
        self.loss_function = loss_function
        self.temperature = temperature
        self.align_params = params.pop(
            'align_params') if 'align_params' in params else None
        if self.align_params is not None:
            for attr, value in self.align_params.items():
                setattr(self, attr, value)

        self.loss_func = BASIC_LOSS.get(loss_function)(**params)

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.layers_name != None:
                assert len(self.layers_name
                           ) == 2, "length of layers_name must be equal to 2."
                out1 = out1[self.layers_name[0]]
                out2 = out2[self.layers_name[1]]
            if self.temperature != 1.0:
                out1 = out1 / self.temperature
                out2 = out2 / self.temperature
            loss_dict["{}_{}_{}_{}_{}".format(self.loss_function, pair[0], pair[
                1], self.layers_name[0] if self.layers_name != None else "0", \
                self.layers_name[1] if self.layers_name != None else "0")] = self.loss_func(out1, out2)
        return loss_dict
