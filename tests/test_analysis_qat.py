import os
import sys
import unittest
sys.path.append("../")
import paddle
from PIL import Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddleslim.quant.analysis_qat import AnalysisQAT

paddle.enable_static()


class ImageNetDataset(DatasetFolder):
    def __init__(self, path, image_size=224):
        super(ImageNetDataset, self).__init__(path)
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(image_size),
            transforms.Transpose(), normalize
        ])

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.samples)


class AnalysisQATDemo(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AnalysisQATDemo, self).__init__(*args, **kwargs)
        if not os.path.exists('MobileNetV1_infer'):
            os.system(
                'wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
            )
            os.system('tar -xf MobileNetV1_infer.tar')
        if not os.path.exists('ILSVRC2012_data_demo'):
            os.system(
                'wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz'
            )
            os.system('tar -xf ILSVRC2012_data_demo.tar.gz')

    def test_demo(self):
        train_dataset = ImageNetDataset(
            "./ILSVRC2012_data_demo/ILSVRC2012/train/")
        image = paddle.static.data(
            name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
        train_loader = paddle.io.DataLoader(
            train_dataset, feed_list=[image], batch_size=8, return_list=False)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        executor = paddle.static.Executor(place)

        ptq_config = {
            'quantizable_op_type': ["conv2d", "depthwise_conv2d"],
            'weight_quantize_type': 'abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'is_full_quantize': False,
            'batch_size': 8,
            'batch_nums': 10,
        }

        post_training_quantization = PostTrainingQuantization(
            executor=executor,
            data_loader=train_loader,
            model_dir="./MobileNetV1_infer",
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            onnx_format=True,
            algo='avg',
            **ptq_config)
        post_training_quantization.quantize()
        post_training_quantization.save_quantized_model(
            "./MobileNetV1_QAT",
            model_filename='inference.pdmodel',
            params_filename='inference.pdiparams')

        analyzer = AnalysisQAT(
            float_model_dir="./MobileNetV1_infer",
            quant_model_dir="./MobileNetV1_QAT",
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            save_dir="MobileNetV1_analysis",
            data_loader=train_loader)
        analyzer.metric_error_analyse()
        os.system('rm -rf MobileNetV1_analysis')
        os.system('rm -rf MobileNetV1_QAT')


if __name__ == '__main__':
    unittest.main()
