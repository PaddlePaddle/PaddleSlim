import os
import unittest
import paddle
from PIL import Image
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import transforms
from paddleslim.auto_compression import AutoCompression
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


class ACTDemo(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ACTDemo, self).__init__(*args, **kwargs)
        os.system(
            'wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar'
        )
        os.system('tar -xf MobileNetV1_infer.tar')
        os.system(
            'wget https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz'
        )
        os.system('tar -xf ILSVRC2012_data_demo.tar.gz')

    def test_demo(self):
        train_dataset = ImageNetDataset(
            "./ILSVRC2012_data_demo/ILSVRC2012/train/")
        image = paddle.static.data(
            name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
        train_loader = paddle.io.DataLoader(
            train_dataset, feed_list=[image], batch_size=32, return_list=False)

        ac = AutoCompression(
            model_dir="./MobileNetV1_infer",
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            save_dir="MobileNetV1_quant",
            config={
                'Quantization': {},
                "HyperParameterOptimization": {
                    'ptq_algo': ['avg'],
                    'max_quant_count': 3
                }
            },
            train_dataloader=train_loader,
            eval_dataloader=train_loader)
        ac.compress()


if __name__ == '__main__':
    unittest.main()
