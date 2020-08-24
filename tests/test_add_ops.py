# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("../")
import unittest
import paddle.fluid as fluid
from paddleslim.prune import Pruner
from paddleslim.core import GraphWrapper
from paddleslim.prune import conv2d as conv2d_walker
from layers import conv_bn_layer


class TestPrune(unittest.TestCase):
    def test_prune(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        #   X       X              O       X              O
        # conv1-->conv2-->sum1-->conv3-->conv4-->sum2-->align_out-->conv5
        #     |            ^ |                    ^   |
        #     |____________| |____________________|    ->gather_out-->conv6
        #                                             |
        #                                              ->lodset_out-->conv7
        #
        # X: prune output channels
        # O: prune input channels
        with fluid.program_guard(main_program, startup_program):
            input = fluid.data(name="image", shape=[None, 3, 64, 64])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            sum1 = conv1 + conv2
            conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
            conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
            sum2 = conv4 + sum1
            #test roi_align
            rois = fluid.data(
                name='rois', shape=[None, 4], dtype='float32')
            align_out = fluid.layers.roi_align(input=sum2,
                                               rois=rois,
                                               pooled_height=7,
                                               pooled_width=7,
                                               spatial_scale=0.5,
                                               sampling_ratio=-1)
            conv5 = conv_bn_layer(align_out, 8, 3, "conv5")
            #test gather
            index = fluid.layers.data(name='index', shape=[-1, 1], dtype='int32')
            gather_out = fluid.layers.gather(sum2, index)
            conv6 = conv_bn_layer(gather_out, 8, 3, "conv6")
            #test lod_reset
            y = fluid.layers.data(name='y', shape=[6], lod_level=2)
            lodset_out = fluid.layers.lod_reset(x=sum2, y=y)
            conv7 = conv_bn_layer(lodset_out, 8, 3, "conv7")

        shapes = {}
        for param in main_program.global_block().all_parameters():
            shapes[param.name] = param.shape

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        exe.run(startup_program, scope=scope)

        graph = GraphWrapper(main_program)

        conv_op = graph.var("conv4_weights").outputs()[0]
        walker = conv2d_walker(conv_op, [])
        walker.prune(graph.var("conv4_weights"), pruned_axis=0, pruned_idx=[])
        print(walker.pruned_params)


if __name__ == '__main__':
    unittest.main()