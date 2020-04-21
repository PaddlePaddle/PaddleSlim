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

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
]

DARTS_V1 = Genotype(
    normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2),
            ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2),
            ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 3),
            ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)],
    reduce_concat=range(2, 6))

DARTS_V2 = Genotype(
    normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 0), ('sep_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_5x5', 3),
            ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=range(2, 6))

PC_DARTS = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0),
            ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2),
            ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 2),
            ('skip_connect', 2), ('avg_pool_3x3', 0)],
    reduce_concat=range(2, 6))
