import os
import sys
import unittest
import numpy as np
sys.path.append("../../")
import paddle
from PIL import Image
from paddleslim.auto_compression import AutoCompression
paddle.enable_static()


class ImageNetDataset(paddle.vision.datasets.DatasetFolder):
    def __init__(self, data_dir, image_size=224, mode='train'):
        super(ImageNetDataset, self).__init__(data_dir)
        self.data_dir = data_dir
        normalize = paddle.vision.transforms.transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        self.transform = paddle.vision.transforms.transforms.Compose([
            paddle.vision.transforms.transforms.Resize(256),
            paddle.vision.transforms.transforms.CenterCrop(image_size),
            paddle.vision.transforms.transforms.Transpose(), normalize
        ])
        self.mode = mode
        train_file_list = os.path.join(data_dir, 'train_list.txt')
        val_file_list = os.path.join(data_dir, 'val_list.txt')
        self.mode = mode
        if mode == 'train':
            with open(train_file_list) as flist:
                full_lines = [line.strip() for line in flist]
                np.random.shuffle(full_lines)
                lines = full_lines
            self.samples = [line.split() for line in lines]
        else:
            with open(val_file_list) as flist:
                lines = [line.strip() for line in flist]
                self.samples = [line.split() for line in lines]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if self.mode == 'train':
            return self.transform(
                Image.open(os.path.join(self.data_dir, img_path)).convert(
                    'RGB'))
        else:
            return self.transform(
                Image.open(os.path.join(self.data_dir, img_path)).convert(
                    'RGB')), np.array([label]).astype('int64')

    def __len__(self):
        return len(self.samples)


class ACTEvalFunction(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ACTEvalFunction, self).__init__(*args, **kwargs)
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
        train_dataset = ImageNetDataset("./ILSVRC2012_data_demo/ILSVRC2012/")
        image = paddle.static.data(
            name='inputs', shape=[None] + [3, 224, 224], dtype='float32')
        label = paddle.static.data(
            name='labels', shape=[None] + [1], dtype='float32')
        train_loader = paddle.io.DataLoader(
            train_dataset, feed_list=[image], batch_size=32, return_list=False)

        def reader_wrapper(reader, input_name):
            def gen():
                for i, (imgs, label) in enumerate(reader()):
                    yield {input_name: imgs}

            return gen

        def eval_reader(data_dir,
                        batch_size,
                        crop_size,
                        resize_size,
                        place=None):
            val_dataset = ImageNetDataset(
                "./ILSVRC2012_data_demo/ILSVRC2012/", mode='val')
            val_loader = paddle.io.DataLoader(
                val_dataset,
                feed_list=[image, label],
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                return_list=False)
            return val_loader

        def eval_function(exe, compiled_test_program, test_feed_names,
                          test_fetch_list):
            val_loader = eval_reader(
                './ILSVRC2012_data_demo/ILSVRC2012/',
                batch_size=32,
                crop_size=224,
                resize_size=256)

            results = []
            print('Evaluating...')
            for batch_id, data in enumerate(val_loader):
                image = data[0]['inputs']
                label = data[0]['labels']
                # top1_acc, top5_acc
                if len(test_feed_names) == 1:
                    image = np.array(image)
                    label = np.array(label).astype('int64')
                    pred = exe.run(compiled_test_program,
                                   feed={test_feed_names[0]: image},
                                   fetch_list=test_fetch_list)
                    pred = np.array(pred[0])
                    label = np.array(label)
                    sort_array = pred.argsort(axis=1)
                    top_1_pred = sort_array[:, -1:][:, ::-1]
                    top_1 = np.mean(label == top_1_pred)
                    top_5_pred = sort_array[:, -5:][:, ::-1]
                    acc_num = 0
                    for i in range(len(label)):
                        if label[i][0] in top_5_pred[i]:
                            acc_num += 1
                    top_5 = float(acc_num) / len(label)
                    results.append([top_1, top_5])
                else:
                    # eval "eval model", which inputs are image and label, output is top1 and top5 accuracy
                    image = np.array(image)
                    label = np.array(label).astype('int64')
                    result = exe.run(compiled_test_program,
                                     feed={
                                         test_feed_names[0]: image,
                                         test_feed_names[1]: label
                                     },
                                     fetch_list=test_fetch_list)
                    result = [np.mean(r) for r in result]
                    results.append(result)
                if batch_id % 100 == 0:
                    print('Eval iter: ', batch_id)
            result = np.mean(np.array(results), axis=0)
            return result[0]

        ac = AutoCompression(
            model_dir="./MobileNetV1_infer",
            model_filename="inference.pdmodel",
            params_filename="inference.pdiparams",
            save_dir="MobileNetV1_eval_quant",
            config='./qat_dist_train.yaml',
            train_dataloader=train_loader,
            eval_callback=eval_function)
        ac.compress()
        os.system('rm -rf MobileNetV1_eval_quant')


if __name__ == '__main__':
    unittest.main()
