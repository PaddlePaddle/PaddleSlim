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

import inspect
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['Counter', 'init_index', 'functional2layer']


class Counter:
    """
        limit the number of function calls.
    """

    def __init__(self, times=1):
        self.calls = 0
        self.times = times

    def __call__(self, func):
        def counter_wrapper(*args, **kwargs):
            func(*args, **kwargs)
            self.calls += 1
            assert self.calls <= self.times, "function {} only allow to call {} times.".format(
                func.__name__, self.times)

        return counter_wrapper


class FuncWrapper(paddle.nn.Layer):
    """
    """

    def __init__(self, functional):
        super(FuncWrapper, self).__init__()
        self.fn = functional

    def forward(self, *x, **kwargs):
        return self.fn(*x, **kwargs)


def convert_fn(fn):
    def new_fn(*x, **kwargs):
        global global_idx
        model = inspect.currentframe().f_back.f_locals['self']
        ### TODO(ceci3): length of sublayers is 0 means not call a nn.Layer in __init__ function.
        ### this condition maybe not rigorous, need to change it later.
        ### model.training set to False is to avoid only eval student model.
        if len(model.sublayers()) == 0 or model.training == False:
            result = eval('F.origin_{}'.format(fn.__name__))(*x, **kwargs)
            return result
        else:
            if getattr(model, 'wrap_fn_{}_{}'.format(fn.__name__, global_idx),
                       None) == None:
                setattr(model, 'wrap_fn_{}_{}'.format(fn.__name__, global_idx),
                        FuncWrapper(fn))
            result = getattr(model, 'wrap_fn_{}_{}'.format(
                fn.__name__, global_idx))(*x, **kwargs)
            global_idx += 1
            return result

    return new_fn


def init_index():
    global global_idx
    global_idx = 0


@Counter()
def functional2layer():
    """
        Wrap the function in paddle.nn.functional with class inherited from paddle.nn.Layer. 
        The purpose of this operation is to get the output of paddle.nn.functional in the model.
        For example:
        ```python
            class Model(nn.Layer):
                def __init__(self):
                    self.fc = nn.Linear(12, 16)
                def forward(x):
                    softmax_out = nn.functional.softmax(x)
                    fc_out = self.fc(softmax_out)
                    relu_out = nn.functional.relu(fc_out)
                    return relu_out
        ```
        Before call the ```paddleslim.common.functional2layer``` function, we can get the output 
        of this model and the output of self.fc function through ```register_forward_post_hook``` in the paddle. 
        And the ```register_forward_post_hook``` interface can only used to get the output of class which is 
        inherited from paddle.nn.Layer. Please reference to: 
        [register_forward_post_hook](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#register_forward_post_hook)
        After call the ```paddleslim.common.functional2layer```, the model above will be converted to
        a new model:
        ```python
            class Model(nn.Layer):
                def __init__(self):
                    self.fc = nn.Linear(12, 16)
                    self.wrap_fn_softmax_0 = FuncWrapper()
                    self.wrap_fn_relu_1 = FuncWrapper()
                def forward(x):
                    softmax_out = self.wrap_fn_softmax_0(x)
                    fc_out = self.fc(softmax_out)
                    relu_out = self.wrap_fn_relu_1(fc_out)
                    return relu_out
        ```
        after this convert operation, we can get the output of softmax through ```register_forward_post_hook```.
        The convert operation can applies to the layers in paddle.nn.functional.py.
    """
    init_index()
    not_convert = ['linear', 'conv1d', 'conv1d_transpose', \
                   'conv2d', 'conv2d_transpose', 'conv3d', \
                   'conv3d_transpose', 'one_hot', 'embedding']
    for f in dir(paddle.nn.functional):
        if not f.startswith('__') and f not in not_convert and not f.startswith(
                'origin_'):
            setattr(paddle.nn.functional, 'origin_{}'.format(f),
                    eval('F.{}'.format(f)))
            if inspect.isfunction(eval('F.{}'.format(f))):
                new_fn = convert_fn(eval('F.{}'.format(f)))
                setattr(paddle.nn.functional, '{}'.format(f), new_fn)
