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

import random
from collections import namedtuple
from ...analysis.flops import dygraph_flops
from ...analysis.latency import TableLatencyEvaluator

ConstraintConfig = namedtuple(
    'ConstraintConfig',
    ['acc_constraint', 'latency_constraint', 'flops_constraint'])
ConstraintConfig.__new__.__defaults__ = (None, ) * len(ConstraintConfig._fields)


class BaseNetConfig:
    def __init__(self):
        raise NotImplementedError("NotImplemented")

    def random_choice(self):
        raise NotImplementedError("NotImplemented")


class EvolutionSearch:
    def __init__(self, net_config, constraint, strategy='EVO', **kwargs):
        assert isinstance(
            constraint,
            ContraintConfig), "constraint must be instance of ContraintConfig"
        assert issubclass(net_config, BaseNetConfig)

        self.net_config = net_config

        if strategy == 'EVO':
            self.strategy = Evolution(**kwargs)
        else:
            raise NotImplementedError("strategy not Implement")

        for key, value in constraint.items():
            setattr(self, key, value)

        self.population_size = getattr(kwargs, 'population_size', 100)
        self.mutate_prob = getattr(kwargs, 'mutate_prob', 0.1)
        self.evo_iter = getattr(kwargs, 'evo_iter', 500)
        self.parent_ratio = getattr(kwargs, 'parent_ratio', 0.25)
        self.mutation_ratio = getattr(kwargs, 'mutation_ratio', 0.5)

        if self.acc_constraint != None:
            input_dim = getattr(self.acc_constraint, 'input_dim', 128)
            pred_model = getattr(self.acc_constraint, 'pred_model', None)
            self.acc_predicter = AccuracyEvaluator(pred_model, input_dim)
            self.min_acc = getattr(self.acc_constraint, 'min_acc', 1.0)

        if self.latency_constraint != None:
            table_file = getattr(self.latency_constraint, 'table_file', None)
            assert table_file != None
            self.latency_predicter = TableLatencyEvaluator(table_file)
            self.max_latency = getattr(self.latency_constraint, 'max_latency',
                                       -1)

        if self.flops_constraint != None:
            self.flops_predicter = dygraph_flops

    def start_search(self):
        mutation_size = int(round(self.population_size * self.mutation_ratio))
        parents_size = int(round(self.population_size * self.parent_ratio))
        best_valid = [-100]

        population = self.random_sample(self.population_size)
        for i in range(self.evo_iter):
            pass

    def satify_constraint(self, sample):
        status = {}
        if self.acc_constraint != None:
            cur_acc = self.acc_predicter(sample)
            if cur_acc < self.min_acc:
                return False, None
            status['acc'] = cur_acc

        if self.latency_constraint != None:
            net = self.convert_onehot_to_net(sample)
            cur_latency = self.latency_predicter.latency(net)
            if cur_latency < self.max_latency:
                return False, None
            status['latency'] = cur_latency

        if self.flops_constraint != None:
            net = self.convert_onehot_to_net(sample)
            cur_flops = self.flops_predicter(net)
            if cur_flops > self.flops_constraint:
                return False, None
            status['flops'] = cur_flops

        return True, status

    def random_sample(self, sample_size=1):
        population = []
        while len(population) < sample_size:
            sample = self.net_config.random_choice()
            satify, constraint_status = self.satify_constraint(sample)
            if satify:
                population.append((sample, constraint_status))

        return population

    def mutate_sample(self, sample):
        pass

    def crossover_sample(self, sample1, sample2):
        pass
