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
import yaml
import inspect
from inspect import isfunction, currentframe
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['config2yaml', 'init_index', 'paddle_convert_fn']


def init_index():
    global global_idx
    global_idx = 0


class counter:
    def __init__(self, times=1):
        self.calls = 0
        self.times = times

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            self.calls += 1
            assert self.calls <= self.times, "function paddle_convert_fn only allow to call {} times.".format(
                self.times)

        return wrapper


class wrapper(nn.Layer):
    def __init__(self, functional):
        super(wrapper, self).__init__()
        self.fn = functional

    def forward(self, *x, **kwargs):
        return self.fn(*x, **kwargs)


def convert_fn(fn):
    def new_fn(*x, **kwargs):
        global global_idx
        model = currentframe().f_back.f_locals['self']
        ### length of sublayers if 0 means is basic layer of paddlepaddle.
        if len(model.sublayers()) == 0:
            result = eval('F.before_{}'.format(fn.__name__))(*x, **kwargs)
            return result
        else:
            if getattr(model, 'wrap_fn_{}_{}'.format(fn.__name__, global_idx),
                       None) == None:
                setattr(model, 'wrap_fn_{}_{}'.format(fn.__name__, global_idx),
                        wrapper(fn))
            result = getattr(model, 'wrap_fn_{}_{}'.format(
                fn.__name__, global_idx))(*x, **kwargs)
            global_idx += 1
            return result

    return new_fn


@counter()
def paddle_convert_fn():
    init_index()
    not_convert = ['linear', 'conv1d', 'conv1d_transpose', \
                   'conv2d', 'conv2d_transpose', 'conv3d', \
                   'conv3d_transpose', 'one_hot', 'embedding']
    print("convert function to class")
    for f in dir(F):
        if not f.startswith('__') and f not in not_convert and not f.startswith(
                'before_') and not f.startswith('after_'):
            setattr(F, 'before_{}'.format(f), eval('F.{}'.format(f)))
            if isfunction(eval('F.{}'.format(f))):
                new_fn = convert_fn(eval('F.{}'.format(f)))
                setattr(F, '{}'.format(f), new_fn)


def yaml2config(yaml_path):
    """
        convert yaml to dict config.
    """
    final_configs = []
    f = open(yaml_path, 'r')
    origin_configs = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    for configs in origin_configs:
        configs = configs['DistillConfig']
        final_configs.extend(configs)
    return final_configs


def config2yaml(configs, yaml_path):
    """
        convert dict config to yaml.
    """
    final_yaml = dict()
    final_yaml['DistillConfig'] = configs
    f = open(yaml_path, "w")
    yaml.dump([final_yaml], f)
    f.close()
