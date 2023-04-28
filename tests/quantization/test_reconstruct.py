import sys
sys.path.append("../../")

import os
import unittest
import paddle
import tempfile
import numpy as np

from paddle.vision.models import resnet18
from paddle.quantization import QuantConfig
from paddleslim.quant.observers import AVGObserver, AbsMaxChannelWiseWeightObserver, ReconstructActObserver, ReconstructWeightObserver, ReconstructPTQ
from paddleslim.quant.observers.avg import AVGObserverLayer
from paddleslim.quant.observers.abs_max_weight import AbsMaxChannelWiseWeightObserverLayer
from paddleslim.quant.observers.reconstruct_act import ReconstructActObserverLayer
from paddleslim.quant.observers.reconstruct_weight import ReconstructWeightObserverLayer


def count_layers(model, layer_type):
    count = 0
    for _layer in model.sublayers(True):
        if isinstance(_layer, layer_type):
            count += 1
    return count


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples, image_shape=[3, 224, 224], class_num=1000):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.class_num = class_num

    def __getitem__(self, idx):
        image = np.random.random(self.image_shape).astype('float32')
        return image

    def __len__(self):
        return self.num_samples


def reader_wrapper(reader):
    def gen():
        for data in reader():
            img = paddle.to_tensor(data)
            yield {'x': img}

    return gen


class TestRestructPTQ(unittest.TestCase):
    def __init__(self, *args, **kvargs):
        super(TestRestructPTQ, self).__init__(*args, **kvargs)

    def setUp(self):
        paddle.set_device("cpu")
        self.temp_dir = tempfile.TemporaryDirectory(dir="./")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_recon_layer(self):
        place = paddle.CUDAPlace(
            0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        model = resnet18()
        weight_layers = self._count_layers(model, paddle.nn.Conv2D)
        weight_layers += self._count_layers(model, paddle.nn.Linear)

        def reader_wrapper(reader, inputs='x'):
            def gen():
                for data in reader():
                    img = paddle.to_tensor(data)
                    yield {inputs: img}

            return gen

        dataset = RandomDataset(100)
        train_reader = paddle.io.DataLoader(
            dataset,
            drop_last=True,
            places=place,
            batch_size=16,
            return_list=True)
        train_loader = reader_wrapper(train_reader)

        # 1. set QuantConfig
        weight_observer = ReconstructWeightObserver(
            ptq_observer=AbsMaxChannelWiseWeightObserver())
        act_observer = ReconstructActObserver(ptq_observer=AVGObserver())
        self.q_config = QuantConfig(
            activation=act_observer, weight=weight_observer)

        # 2. initialize ReconstructPTQ
        recon_ptq = ReconstructPTQ(
            model,
            self.q_config,
            train_loader,
            epochs=1,
            batch_nums=10,
            lr=0.1,
            recon_level='layer-wise')

        recon_ptq.init_ptq()

        # 3. run ReconstructPTQ
        quant_model = recon_ptq.run()

        quantizer_cnt = self._count_layers(quant_model,
                                           ReconstructWeightObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model,
                                           ReconstructActObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model,
                                           AbsMaxChannelWiseWeightObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model, AVGObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)

        # 4. save quantized model
        final_model = recon_ptq.convert(quant_model, True)
        inputs = paddle.static.InputSpec(
            [None, 3, 224, 224], 'float32', name='x')
        paddle.jit.save(final_model, self.temp_dir + '/recon_layer', [inputs])

    def test_recon_region(self):
        place = paddle.CUDAPlace(
            0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        model = resnet18()
        weight_layers = self._count_layers(model, paddle.nn.Conv2D)
        weight_layers += self._count_layers(model, paddle.nn.Linear)

        def reader_wrapper(reader, inputs='x'):
            def gen():
                for data in reader():
                    img = paddle.to_tensor(data)
                    yield {inputs: img}

            return gen

        dataset = RandomDataset(100)
        train_reader = paddle.io.DataLoader(
            dataset,
            drop_last=True,
            places=place,
            batch_size=16,
            return_list=True)
        train_loader = reader_wrapper(train_reader)

        # 1. set QuantConfig
        weight_observer = ReconstructWeightObserver(
            ptq_observer=AbsMaxChannelWiseWeightObserver())
        act_observer = ReconstructActObserver(
            ptq_observer=AVGObserver(), qdrop=True)
        self.q_config = QuantConfig(
            activation=act_observer, weight=weight_observer)

        # 2. initialize ReconstructPTQ
        recon_ptq = ReconstructPTQ(
            model,
            self.q_config,
            train_loader,
            epochs=1,
            batch_nums=10,
            lr=0.1,
            recon_level='region-wise')

        recon_ptq.init_ptq()

        # 3. run ReconstructPTQ
        quant_model = recon_ptq.run()

        quantizer_cnt = self._count_layers(quant_model,
                                           ReconstructWeightObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model,
                                           ReconstructActObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model,
                                           AbsMaxChannelWiseWeightObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model, AVGObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)

        # 4. save quantized model
        final_model = recon_ptq.convert(quant_model, True)
        inputs = paddle.static.InputSpec(
            [None, 3, 224, 224], 'float32', name='x')
        paddle.jit.save(final_model, self.temp_dir + '/recon_region', [inputs])


class TestRestructPTQ2(TestRestructPTQ):
    def __init__(self, *args, **kvargs):
        super(TestRestructPTQ2, self).__init__(*args, **kvargs)

    def test_recon_region(self):
        place = paddle.CUDAPlace(
            0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        model = resnet18()
        weight_layers = self._count_layers(model, paddle.nn.Conv2D)
        weight_layers += self._count_layers(model, paddle.nn.Linear)

        def reader_wrapper(reader, inputs='x'):
            def gen():
                for data in reader():
                    img = paddle.to_tensor(data)
                    yield {inputs: img}

            return gen

        dataset = RandomDataset(100)
        train_reader = paddle.io.DataLoader(
            dataset,
            drop_last=True,
            places=place,
            batch_size=16,
            return_list=True)
        train_loader = reader_wrapper(train_reader)

        # 1. set QuantConfig
        weight_observer = ReconstructWeightObserver(
            ptq_observer=AbsMaxChannelWiseWeightObserver())
        act_observer = ReconstructActObserver(
            ptq_observer=AVGObserver(), qdrop=True)
        self.q_config = QuantConfig(
            activation=act_observer, weight=weight_observer)

        # 2. initialize ReconstructPTQ
        recon_ptq = ReconstructPTQ(
            model,
            self.q_config,
            train_loader,
            epochs=2,
            batch_nums=15,
            lr=0.1,
            recon_level='region-wise')

        recon_ptq.init_ptq()

        # 3. run ReconstructPTQ
        quant_model = recon_ptq.run()

        quantizer_cnt = self._count_layers(quant_model,
                                           ReconstructWeightObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model,
                                           ReconstructActObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model,
                                           AbsMaxChannelWiseWeightObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)
        quantizer_cnt = self._count_layers(quant_model, AVGObserverLayer)
        self.assertEqual(quantizer_cnt, weight_layers)

        # 4. save quantized model
        final_model = recon_ptq.convert(quant_model, True)
        inputs = paddle.static.InputSpec(
            [None, 3, 224, 224], 'float32', name='x')
        paddle.jit.save(final_model, self.temp_dir + '/recon_region', [inputs])


if __name__ == '__main__':
    unittest.main()
