# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def get_genotype(model):
    def _parse(weights, weights2=None):
        gene = []
        n = 2
        start = 0
        for i in range(model._steps):
            end = start + n
            W = weights[start:end].copy()
            if model._method == "PC-DARTS":
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != model._primitives.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != model._primitives.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((model._primitives[k_best], j))
            start = end
            n += 1
        return gene

    weightsr2 = None
    weightsn2 = None

    gene_normal = _parse(
        paddle.nn.functional.softmax(model.alphas_normal).numpy(), weightsn2)
    gene_reduce = _parse(
        paddle.nn.functional.softmax(model.alphas_reduce).numpy(), weightsr2)

    concat = range(2 + model._steps - model._multiplier, model._steps + 2)
    genotype = Genotype(
        normal=gene_normal,
        normal_concat=concat,
        reduce=gene_reduce,
        reduce_concat=concat)
    return genotype
