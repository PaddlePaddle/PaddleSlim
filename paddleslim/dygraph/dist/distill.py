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

__all__ = ['DistillConfig', 'Distill']

DistillConfig = namedtuple(
    'DistillConfig',
    [
        # list of instance of model, instance of student model, default: None.
        'student_models(model|list(model))',
        # list of instance of model, instance of teacher model, default: None.
        'teacher_models(model|list(model))',
        # list(dict)
        'distill_layers'
    ])
DistillConfig.__new__.__defaults__ = (None, ) * len(DistillConfig._fields)


class Distill(nn.Layer):
    def __init__(self, distill_config):
        self._distill_config = distill_config
        distill_layers = self._distill_config.distill_layers
        assert (self.distill_layers != None)
        self.student_mapping_layers = []
        self.teacher_mapping_layers = []
        self.loss_config = {}
        for distill_layer in distill_layers:
            Sname = distill_layer['student_layer_name']
            Tname = distill_layer['teacher_layer_name']
            loss_funcs = distill_layer['loss_function']
            weights = distill_layer['weight']
            aligns = distill_layer['align']
            self.student_mapping_layers.append(Sname)
            self.teacher_mapping_layers.append(Tname)
            for idx, (loss_func, weight,
                      align) in enumerate(zip(loss_funcs, weights, aligns)):
                ### key: unique name; value: loss_function, weight, 
                ### align_config(op, shape), idx(the number of output in this layer)
                self.loss_config[Sname + '\0' + Tname + '\0' + loss_func +
                                 '\0'] = (loss_func, weight, align, idx)
        self._prepare_featuremap()

    def _prepare_featuremap(self):
        self.Tacts, self.Sacts = {}, {}
        self.hooks = []

        def get_activation(mem, name):
            def get_output_hook(layer, input, output):
                mem[name] = output

            return get_output_hook

        def add_hook(net, mem, mapping_layers):
            for idx, (n, m) in enumerate(net.named_sublayers()):
                if n in mapping_layers:
                    self.hooks.append(
                        m.register_forward_post_hook(get_activation(mem, n)))
            add_hook(self.student_model, self.Sacts,
                     self.student_mapping_layers)
            add_hook(self.teacher_model, self.Tacts,
                     self.teacher_mapping_layers)
