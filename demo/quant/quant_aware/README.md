# 在线量化示例

本示例介绍如何使用在线量化接口，来对训练好的分类模型进行量化, 可以减少模型的存储空间和显存占用。

## 接口介绍

请参考 <a href='https://paddlepaddle.github.io/PaddleSlim/api_cn/quantization_api.html#quant-aware'>量化API文档</a>。

## 分类模型的量化训练流程

### 准备数据

在``demo``文件夹下创建``data``文件夹，将``ImageNet``数据集解压在``data``文件夹下，解压后``data/ILSVRC2012``文件夹下应包含以下文件：
- ``'train'``文件夹，训练图片
- ``'train_list.txt'``文件
- ``'val'``文件夹，验证图片
- ``'val_list.txt'``文件

### 准备需要量化的模型

使用以下命令下载训练好的模型并解压。

```
mkdir pretrain
cd pretrain
wget http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar xf MobileNetV1_pretrained.tar
cd ..
```

### 配置量化参数

```
quant_config = {
    'weight_quantize_type': 'abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'not_quant_pattern': ['skip_quant'],
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9
}
```

### 对训练和测试program插入可训练量化op

```
val_program = quant_aware(val_program, place, quant_config, scope=None, for_test=True)

compiled_train_prog = quant_aware(train_prog, place, quant_config, scope=None, for_test=False)
```

### 关掉指定build策略

```
build_strategy = paddle.static.BuildStrategy()
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
exec_strategy = paddle.static.ExecutionStrategy()
compiled_train_prog = compiled_train_prog.with_data_parallel(
        loss_name=avg_cost.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
```


### 训练命令

```
python train.py --model MobileNet --pretrained_model ./pretrain/MobileNetV1_pretrained --checkpoint_dir ./output/mobilenetv1 --num_epochs 30
```
运行之后，可看到``best_model``的最后测试结果，和MobileNet量化前的精度top1=70.99%, top5=89.68%非常相近。
