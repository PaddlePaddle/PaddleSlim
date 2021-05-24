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
sys.path.append("../../")
import unittest
import time
import numpy as np
import paddle
import paddle.fluid as fluid
from paddleslim.prune import sensitivity
import paddle.vision.transforms as T
from paddle.static import InputSpec as Input
from paddleslim.dygraph import L1NormFilterPruner


class TestSensitivity(unittest.TestCase):
    def __init__(self, methodName='runTest', pruner=None, param_names=[]):
        super(TestSensitivity, self).__init__(methodName)
        self._pruner = pruner
        self._param_names = param_names
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        self.train_dataset = paddle.vision.datasets.MNIST(
            mode="train", backend="cv2", transform=transform)
        self.val_dataset = paddle.vision.datasets.MNIST(
            mode="test", backend="cv2", transform=transform)

        def _reader():
            for data in self.val_dataset:
                yield data

        self.val_reader = _reader

    def runTest(self):
        dygraph_sen, params = self.dygraph_sen()
        static_sen = self.static_sen(params)
        all_right = True
        for _name, _value in dygraph_sen.items():
            _losses = {}
            for _ratio, _loss in static_sen[_name].items():
                _losses[round(_ratio, 2)] = _loss
            for _ratio, _loss in _value.items():
                if not np.allclose(_losses[_ratio], _loss, atol=1e-2):
                    print(
                        f'static loss: {static_sen[_name][_ratio]}; dygraph loss: {_loss}'
                    )
                    all_right = False
        self.assertTrue(all_right)

    def dygraph_sen(self):
        paddle.disable_static()
        net = paddle.vision.models.LeNet()
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=net.parameters())
        inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]
        model = paddle.Model(net, inputs, labels)
        model.prepare(
            optimizer,
            paddle.nn.CrossEntropyLoss(),
            paddle.metric.Accuracy(topk=(1, 5)))

        model.fit(self.train_dataset, epochs=1, batch_size=128, verbose=1)
        result = model.evaluate(self.val_dataset, batch_size=128, verbose=1)
        pruner = None
        if self._pruner == 'l1norm':
            pruner = L1NormFilterPruner(net, [1, 1, 28, 28])
        elif self._pruner == 'fpgm':
            pruner = FPGMFilterPruner(net, [1, 1, 28, 28])

        def eval_fn():
            result = model.evaluate(self.val_dataset, batch_size=128)
            return result['acc_top1']

        sen = pruner.sensitive(
            eval_func=eval_fn,
            sen_file="_".join(["./dygraph_sen_", str(time.time())]),
            #sen_file="sen.pickle",
            target_vars=self._param_names)
        params = {}
        for param in net.parameters():
            params[param.name] = np.array(param.value().get_tensor())
        print(f'dygraph sen: {sen}')
        return sen, params

    def static_sen(self, params):
        paddle.enable_static()
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                input = fluid.data(name="image", shape=[None, 1, 28, 28])
                label = fluid.data(name="label", shape=[None, 1], dtype="int64")
                model = paddle.vision.models.LeNet()
                out = model(input)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        eval_program = main_program.clone(for_test=True)
        place = fluid.CUDAPlace(0)
        scope = fluid.global_scope()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        val_reader = paddle.fluid.io.batch(self.val_reader, batch_size=128)

        def eval_func(program):
            feeder = fluid.DataFeeder(
                feed_list=['image', 'label'], place=place, program=program)
            acc_set = []
            for data in val_reader():
                acc_np = exe.run(program=program,
                                 feed=feeder.feed(data),
                                 fetch_list=[acc_top1])
                acc_set.append(float(acc_np[0]))
            acc_val_mean = np.array(acc_set).mean()
            return acc_val_mean

        for _name, _value in params.items():
            t = scope.find_var(_name).get_tensor()
            t.set(_value, place)
        print(f"static base: {eval_func(eval_program)}")
        criterion = None
        if self._pruner == 'l1norm':
            criterion = 'l1_norm'
        elif self._pruner == 'fpgm':
            criterion = 'geometry_median'
        sen = sensitivity(
            eval_program,
            place,
            self._param_names,
            eval_func,
            sensitivities_file="_".join(
                ["./sensitivities_file", str(time.time())]),
            criterion=criterion)
        return sen


def add_cases(suite):
    suite.addTest(
        TestSensitivity(
            pruner="l1norm", param_names=["conv2d_0.w_0"]))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
