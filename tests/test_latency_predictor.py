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
import sys, os
sys.path.append("../")
import unittest
import paddle
import paddleslim
from paddleslim.analysis import LatencyPredictor, TableLatencyPredictor
from paddle.vision.models import mobilenet_v1, mobilenet_v2
from paddle.nn import Conv2D, BatchNorm2D, ReLU, LayerNorm
from paddleslim.analysis._utils import opt_model, save_cls_model, save_det_model


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups

    x = paddle.reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])

    x = paddle.transpose(x=x, perm=[0, 2, 1, 3, 4])

    x = paddle.reshape(x=x, shape=[batch_size, num_channels, height, width])
    return x


class ModelCase1(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase1, self).__init__()
        self.conv1 = Conv2D(58, 58, 1)
        self.conv2 = Conv2D(58, 58, 1)

    def forward(self, inputs):
        x1, x2 = paddle.split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        out = paddle.concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class ModelCase2(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase2, self).__init__()
        self.conv1 = Conv2D(3, 24, 3, stride=2, padding=1)

    def forward(self, inputs):
        image = inputs['image']

        return self.conv1(image)


class ModelCase3(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase3, self).__init__()
        self.conv1 = Conv2D(3, 24, 3, stride=2, padding=1)

    def forward(self, inputs):
        image = inputs['image']
        im_shape = inputs['im_shape']
        scale_factor = inputs['scale_factor']

        return self.conv1(image), im_shape, scale_factor


class ModelCase4(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase4, self).__init__()
        self.bn1 = BatchNorm2D(3)
        self.ln1 = LayerNorm([3 * 16 * 16])
        self.relu1 = ReLU()
        self.fc1 = paddle.nn.Linear(3 * 16 * 16, 3 * 16 * 16)

    def forward(self, inputs):
        x = self.bn1(inputs)
        x = paddle.reshape(x, [1, 3 * 16 * 16])
        x = self.ln1(x)
        x = self.fc1(x)
        x = paddle.fluid.layers.unsqueeze(input=x, axes=[2])
        x = self.relu1(x)
        y = paddle.fluid.layers.fill_constant(
            x.shape, dtype=paddle.float32, value=1)
        x = paddle.stack([x, y], axis=3)
        x = paddle.slice(x, axes=[0], starts=[0], ends=[1])
        x = paddle.exp(x)
        y += paddle.fluid.layers.uniform_random(y.shape)
        y = paddle.fluid.layers.reduce_mean(y, dim=1, keep_dim=True)
        return paddle.greater_equal(x, y)


class ModelCase5(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase5, self).__init__()
        self.bn1 = BatchNorm2D(255)

    def forward(self, inputs):
        image = inputs['image']
        image = self.bn1(image)
        img_size = paddle.fluid.data(
            name='img_size', shape=[None, 2], dtype='int64')
        anchors = [10, 13, 16, 30, 33, 23]
        boxes, scores = paddle.fluid.layers.yolo_box(
            x=image,
            img_size=img_size,
            class_num=80,
            anchors=anchors,
            conf_thresh=0.01,
            downsample_ratio=32)
        out = paddle.fluid.layers.matrix_nms(
            bboxes=boxes,
            scores=scores,
            background_label=0,
            score_threshold=0.5,
            post_threshold=0.1,
            nms_top_k=400,
            keep_top_k=200,
            normalized=False)
        box, var = paddle.fluid.layers.prior_box(
            input=image, image=image, min_sizes=[2.], clip=True, flip=True)
        return boxes, scores, box, var, out


class ModelCase6(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase6, self).__init__()
        self.bn1 = BatchNorm2D(3)
        self.relu1 = ReLU()
        self.fc1 = paddle.nn.Linear(3 * 16 * 16, 3 * 16 * 16)
        self.dp = paddle.nn.Dropout(p=0.5)
        self.lstm = paddle.nn.LSTM(
            1536, 10, direction='bidirectional', num_layers=2)

    def forward(self, inputs):
        x = self.bn1(inputs)
        x = paddle.reshape(x, [1, 3 * 16 * 16])
        x = self.fc1(x)
        x = paddle.fluid.layers.unsqueeze(input=x, axes=[2])
        x = self.relu1(x)
        y = paddle.fluid.layers.fill_constant(
            x.shape, dtype=paddle.float32, value=1)
        # x = paddle.stack([x, y], axis=3)
        x = paddle.slice(x, axes=[0], starts=[0], ends=[1])
        x = paddle.exp(x)
        # y += paddle.fluid.layers.uniform_random(y.shape)
        y = paddle.expand(y, shape=[1, 768, 768, 2])
        x = paddle.expand(x, shape=[1, 768, 768, 2])
        out = paddle.concat([x, y])
        out = self.dp(out)
        out = channel_shuffle(out, 2)
        out1, out2 = paddle.split(out, num_or_sections=2, axis=1)
        outshape = out1.shape
        max_idx = paddle.argmax(
            out1.reshape((outshape[0], outshape[1], outshape[2] * outshape[3])),
            axis=-1)
        out2 = out2.reshape(
            (outshape[0], outshape[1], outshape[2] * outshape[3]))
        res, _ = self.lstm(out2)
        return res, max_idx


class ModelCase7(paddle.nn.Layer):
    def __init__(self):
        super(ModelCase7, self).__init__()
        self.bn1 = BatchNorm2D(255)

    def forward(self, inputs):
        image = inputs['image']
        image = self.bn1(image)
        img_size = paddle.fluid.data(
            name='img_size', shape=[None, 2], dtype='int64')
        anchors = [10, 13, 16, 30, 33, 23]
        boxes, scores = paddle.fluid.layers.yolo_box(
            x=image,
            img_size=img_size,
            class_num=80,
            anchors=anchors,
            conf_thresh=0.01,
            downsample_ratio=32)
        box, var = paddle.fluid.layers.prior_box(
            input=image, image=image, min_sizes=[2.], clip=True, flip=True)
        return boxes, scores, box, var


class TestCase(unittest.TestCase):
    def setUp(slef):
        os.system(
            'wget -q https://bj.bcebos.com/v1/paddle-slim-models/LatencyPredictor/test_mobilenetv1.tar'
        )
        os.system('tar -xf test_mobilenetv1.tar')
        os.system(
            'wget -q https://bj.bcebos.com/v1/paddle-slim-models/LatencyPredictor/test_mobilenetv1_qat.tar'
        )
        os.system('tar -xf test_mobilenetv1_qat.tar')

    def test_case1(self):
        paddle.disable_static()
        predictor = TableLatencyPredictor(table_file='SD710')
        model_file = 'test_mobilenetv1/inference.pdmodel'
        param_file = 'test_mobilenetv1/inference.pdiparams'
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0

        model_file = 'test_mobilenetv1_qat/inference.pdmodel'
        param_file = 'test_mobilenetv1_qat/inference.pdiparams'
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='int8')
        assert latency > 0


class TestCase2(unittest.TestCase):
    def setUp(slef):
        os.system(
            'wget -q https://bj.bcebos.com/v1/paddle-slim-models/LatencyPredictor/test_mobilenetv2.tar'
        )
        os.system('tar -xf test_mobilenetv2.tar')
        os.system(
            'wget -q https://bj.bcebos.com/v1/paddle-slim-models/LatencyPredictor/test_mobilenetv2_qat.tar'
        )
        os.system('tar -xf test_mobilenetv2_qat.tar')

    def _infer_shape(self, model_dir, model_filename, params_filename,
                     input_shapes, save_path):
        assert type(input_shapes) in [
            dict, list, tuple
        ], f'Type of input_shapes should be in [dict, tuple or list] but got {type(input_shapes)}.'
        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())

        model_name = '.'.join(model_filename.split('.')[:-1])
        model_path_prefix = os.path.join(model_dir, model_name)
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(
                path_prefix=model_path_prefix, executor=exe))

        if type(input_shapes) in [list, tuple]:
            assert len(
                feed_target_names
            ) == 1, f"The number of model's inputs should be 1 but got {feed_target_names}."
            input_shapes = {feed_target_names[0]: input_shapes}

        feed_vars = []
        for var_ in inference_program.list_vars():
            if var_.name in feed_target_names:
                feed_vars.append(var_)
                var_.desc.set_shape(input_shapes[var_.name])

        for block in inference_program.blocks:
            for op in block.ops:
                if op.type not in ["feed", "fetch"]:
                    op.desc.infer_shape(block.desc)

        save_path = os.path.join(save_path, "infered_shape")
        os.makedirs(save_path)
        paddle.static.save_inference_model(
            save_path,
            feed_vars,
            fetch_targets,
            exe,
            program=inference_program,
            clip_extra=False)
        print(f"Saved model infered shape to {save_path}")

    def test_case2(self):
        predictor = TableLatencyPredictor(table_file='SD710')
        model_file = 'test_mobilenetv2/inference.pdmodel'
        param_file = 'test_mobilenetv2/inference.pdiparams'
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0

        pbmodel_file = opt_model(
            model_file=model_file,
            param_file=param_file,
            optimize_out_type='protobuf')

        pred = LatencyPredictor()
        paddle.enable_static()
        with open(pbmodel_file, "rb") as f:
            fluid_program = paddle.fluid.framework.Program.parse_from_string(
                f.read())
            graph = paddleslim.core.GraphWrapper(fluid_program)
            graph_keys = pred._get_key_info_from_graph(graph=graph)
            assert len(graph_keys) > 0

        self._infer_shape(
            model_dir='test_mobilenetv2',
            model_filename='inference.pdmodel',
            params_filename='inference.pdiparams',
            input_shape=[1, 3, 250, 250],
            save_path='test_mobilenetv2_250')

        model_file = 'test_mobilenetv2_250/infered_shape.pdmodel'
        param_file = 'test_mobilenetv2_250/infered_shape.pdiparams'
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0

        self._infer_shape(
            model_dir='test_mobilenetv2_qat',
            model_filename='inference.pdmodel',
            params_filename='inference.pdiparams',
            input_shape=[1, 3, 250, 250],
            save_path='test_mobilenetv2_qat_250')

        model_file = 'test_mobilenetv2_qat_250/infered_shape.pdmodel'
        param_file = 'test_mobilenetv2_qat_250/infered_shape.pdiparams'
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='int8')
        assert latency > 0


class TestCase3(unittest.TestCase):
    def test_case3(self):
        paddle.disable_static()
        model = ModelCase1()
        model_file, param_file = save_cls_model(
            model,
            input_shape=[1, 116, 28, 28],
            save_dir="./inference_model",
            data_type='fp32')
        predictor = TableLatencyPredictor(table_file='SD710')
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0


class TestCase4(unittest.TestCase):
    def test_case4(self):
        paddle.disable_static()
        model = ModelCase2()
        predictor = TableLatencyPredictor(table_file='SD710')
        model_file, param_file = save_det_model(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir="./inference_model",
            data_type='fp32')
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0


class TestCase5(unittest.TestCase):
    def test_case5(self):
        paddle.disable_static()
        model = ModelCase3()
        predictor = TableLatencyPredictor(table_file='SD710')
        model_file, param_file = save_det_model(
            model,
            input_shape=[1, 3, 224, 224],
            save_dir="./inference_model",
            data_type='fp32',
            det_multi_input=True)
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0


class TestCase6(unittest.TestCase):
    def test_case6(self):
        paddle.disable_static()
        model = ModelCase4()
        predictor = LatencyPredictor()
        model_file, param_file = save_cls_model(
            model,
            input_shape=[1, 3, 16, 16],
            save_dir="./inference_model",
            data_type='int8')
        pbmodel_file = opt_model(
            model_file=model_file,
            param_file=param_file,
            optimize_out_type='protobuf')

        paddle.enable_static()
        with open(pbmodel_file, "rb") as f:
            fluid_program = paddle.fluid.framework.Program.parse_from_string(
                f.read())
            graph = paddleslim.core.GraphWrapper(fluid_program)
            graph_keys = predictor._get_key_info_from_graph(graph=graph)
            assert len(graph_keys) > 0


class TestCase7(unittest.TestCase):
    def test_case7(self):
        paddle.disable_static()
        model = ModelCase5()
        predictor = LatencyPredictor()
        model_file, param_file = save_det_model(
            model,
            input_shape=[1, 255, 13, 13],
            save_dir="./inference_model",
            data_type='fp32')
        pbmodel_file = opt_model(
            model_file=model_file,
            param_file=param_file,
            optimize_out_type='protobuf')

        paddle.enable_static()
        with open(pbmodel_file, "rb") as f:
            fluid_program = paddle.fluid.framework.Program.parse_from_string(
                f.read())
            graph = paddleslim.core.GraphWrapper(fluid_program)
            graph_keys = predictor._get_key_info_from_graph(graph=graph)
            assert len(graph_keys) > 0


class TestCase8(unittest.TestCase):
    def test_case8(self):
        paddle.disable_static()
        predictor = TableLatencyPredictor(table_file='SD710')
        model = ModelCase6()
        model_file, param_file = save_cls_model(
            model,
            input_shape=[1, 3, 16, 16],
            save_dir="./inference_model",
            data_type='fp32')
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0

        model2 = ModelCase7()
        model_file, param_file = save_det_model(
            model2,
            input_shape=[1, 255, 14, 14],
            save_dir="./inference_model",
            data_type='fp32')
        latency = predictor.predict(
            model_file=model_file, param_file=param_file, data_type='fp32')
        assert latency > 0


if __name__ == '__main__':
    unittest.main()
