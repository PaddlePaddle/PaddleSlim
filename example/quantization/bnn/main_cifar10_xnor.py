import argparse
import os
import random
import numpy as np
import time
import paddle
import paddle.nn as nn
import paddle.vision.transforms  as T
from paddle.io import Dataset, BatchSampler, DataLoader

from binarizer.xnor import XNORConv2d # 以XNOR为例

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class AccLossCallback(paddle.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
        self.epoch_eval_acc = []
        self.epoch_eval_loss = []
        self.epoch_train_acc = []
        self.epoch_train_loss = []
        self.save_path = 'weights/best'
        self.max_acc = None

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_train_loss.append(logs.get('loss')[0])
        self.epoch_train_acc.append(logs.get('acc'))
    

    def on_eval_end(self, logs=None):
        self.epoch_eval_loss.append(logs.get('loss')[0])
        self.epoch_eval_acc.append(logs.get('acc'))
        acc = logs.get('acc')
        if self.max_acc is None or acc > self.max_acc:
            self.max_acc = acc
            self.model.save(self.save_path)
        print('max acc is {}'.format(self.max_acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=1, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='datasets/cifar-10-python.tar.gz', type=str, help='path to Cifar data')

    args = parser.parse_args()

    paddle.device.set_device("gpu:0")
    # paddle.device.set_device("cpu")
    seed_all(args.seed)
    # build imagenet data loader
    transform1=T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(0.5),
            T.Transpose(),
            T.Normalize((125.307, 122.961, 113.8575), (51.5865, 50.847, 51.255))
        ]
    )
    transform2=T.Compose(
        [
            T.Transpose(),
            T.Normalize((125.307, 122.961, 113.8575), (51.5865, 50.847, 51.255))
        ]
    )

    
    train_set = paddle.vision.datasets.Cifar10(data_file=args.data_path, 
                                    mode='train',
                                    transform=transform1)
    val_set = paddle.vision.datasets.Cifar10(data_file=args.data_path, 
                                    mode='test',
                                    transform=transform2)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    test_loader = DataLoader(val_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    from models.resnet import resnet18
    cnn = resnet18(pretrained=False, num_classes=10)
    
    def quant_module_refactor(module: nn.Layer):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2D) and not hasattr(child, 'no_quant'):
                in_channels = child._in_channels
                out_channels = child._out_channels
                kernel_size = child._kernel_size[0]
                stride = child._stride
                padding = child._padding
                bias = child._bias_attr

                new_conv = XNORConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                setattr(module, name, new_conv)
            else:
                quant_module_refactor(child)

    layers = cnn.sublayers()
    setattr(layers[0], 'no_quant', True)
    setattr(layers[-1], 'no_quant', True)

    quant_module_refactor(cnn)

    print(cnn)

    state_dict = paddle.load("weights/best_pretrain.pdparams") # 可以先加载预训练的全精度权重以初始化，非必须
    cnn.set_state_dict(state_dict)
    
    model = paddle.Model(cnn)

    mylogs = AccLossCallback()

    lr = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True) 

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=1e-1, T_max=400, verbose=True)
    optim =  paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9,weight_decay=1e-4)

    model.prepare(optim, loss=paddle.nn.CrossEntropyLoss(),metrics=paddle.metric.Accuracy())
    save_dir = './fp_cifar'
    model.fit(train_loader,test_loader,epochs=400,batch_size=args.batch_size,callbacks=[mylogs, lr],
              verbose=1,num_workers=1,save_dir=save_dir,save_freq=200)
    
    eval_result = model.evaluate(test_loader, verbose=1)
    print(eval_result)
