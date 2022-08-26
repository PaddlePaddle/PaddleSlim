import sys
import os
sys.path.append("../../")
import unittest
import tempfile
import paddle
import unittest
import numpy as np
from paddle.io import Dataset
from paddleslim.auto_compression import AutoCompression
from paddleslim.common import load_config
from paddleslim.common import load_inference_model, export_onnx


class RandomEvalDataset(Dataset):
    def __init__(self, num_samples, image_shape=[3, 32, 32], class_num=10):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.class_num = class_num

    def __getitem__(self, idx):
        image = np.random.random(self.image_shape).astype('float32')
        return image

    def __len__(self):
        return self.num_samples


class ACTBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ACTBase, self).__init__(*args, **kwargs)
        paddle.enable_static()
        self.tmpdir = tempfile.TemporaryDirectory(prefix="test_")
        self.infer_model_dir = os.path.join(self.tmpdir.name, "infer")
        self.create_program()
        self.create_dataloader()

    def create_program(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.static.data(
                name='data', shape=[-1, 3, 32, 32], dtype='float32')
            tmp = paddle.static.nn.conv2d(
                input=data, num_filters=2, filter_size=3)
            out = paddle.static.nn.conv2d(
                input=tmp, num_filters=2, filter_size=3)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(startup_program)

        paddle.static.save_inference_model(
            self.infer_model_dir, [data], [out], exe, program=main_program)
        print(f"saved infer model to [{self.infer_model_dir}]")

    def create_dataloader(self):
        # define a random dataset
        self.eval_dataset = RandomEvalDataset(32)

    def __del__(self):
        self.tmpdir.cleanup()


class TestYamlQATDistTrain(ACTBase):
    def __init__(self, *args, **kwargs):
        super(TestYamlQATDistTrain, self).__init__(*args, **kwargs)

    def test_compress(self):
        image = paddle.static.data(
            name='data', shape=[-1, 3, 32, 32], dtype='float32')
        train_loader = paddle.io.DataLoader(
            self.eval_dataset, feed_list=[image], batch_size=4)
        ac = AutoCompression(
            model_dir=self.tmpdir.name,
            model_filename="infer.pdmodel",
            params_filename="infer.pdiparams",
            save_dir="output",
            config="./qat_dist_train.yaml",
            train_dataloader=train_loader,
            eval_dataloader=train_loader)  # eval_function to verify accuracy
        ac.compress()


class TestSetQATDist(ACTBase):
    def __init__(self, *args, **kwargs):
        super(TestSetQATDist, self).__init__(*args, **kwargs)

    def test_compress(self):
        image = paddle.static.data(
            name='data', shape=[-1, 3, 32, 32], dtype='float32')
        train_loader = paddle.io.DataLoader(
            self.eval_dataset, feed_list=[image], batch_size=4)
        ac = AutoCompression(
            model_dir=self.tmpdir.name,
            model_filename="infer.pdmodel",
            params_filename="infer.pdiparams",
            save_dir="output",
            config={"QAT", "Distillation"},
            train_dataloader=train_loader,
            eval_dataloader=train_loader)  # eval_function to verify accuracy
        ac.compress()


class TestDictQATDist(ACTBase):
    def __init__(self, *args, **kwargs):
        super(TestDictQATDist, self).__init__(*args, **kwargs)

    def test_compress(self):
        config = load_config("./qat_dist_train.yaml")
        image = paddle.static.data(
            name='data', shape=[-1, 3, 32, 32], dtype='float32')
        train_loader = paddle.io.DataLoader(
            self.eval_dataset, feed_list=[image], batch_size=4)
        ac = AutoCompression(
            model_dir=self.tmpdir.name,
            model_filename="infer.pdmodel",
            params_filename="infer.pdiparams",
            save_dir="output",
            config=config,
            train_dataloader=train_loader,
            eval_dataloader=train_loader)  # eval_function to verify accuracy
        ac.compress()


class TestLoadONNXModel(ACTBase):
    def __init__(self, *args, **kwargs):
        super(TestLoadONNXModel, self).__init__(*args, **kwargs)
        os.system(
            'wget -q https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx')
        self.model_dir = 'yolov5s.onnx'

    def test_compress(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        _, _, _ = load_inference_model(
            self.model_dir,
            executor=exe,
            model_filename='model.pdmodel',
            params_filename='model.paiparams')
        # reload model
        _, _, _ = load_inference_model(
            self.model_dir,
            executor=exe,
            model_filename='model.pdmodel',
            params_filename='model.paiparams')
        # convert onnx
        export_onnx(
            self.model_dir,
            model_filename='model.pdmodel',
            params_filename='model.paiparams',
            save_file_path='output.onnx',
            opset_version=13,
            deploy_backend='tensorrt')


if __name__ == '__main__':
    unittest.main()
