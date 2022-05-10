# 非结构化稀疏 -- 静态图剪裁（包括按照阈值和比例剪裁两种模式）示例

## 简介

在模型压缩中，常见的稀疏方式为结构化和非结构化稀疏，前者在某个特定维度（特征通道、卷积核等等）上进行稀疏化操作；后者以每一个参数为单元进行稀疏化，并不会改变参数矩阵的形状，所以更加依赖于硬件对稀疏后矩阵运算的加速能力。本目录即在PaddlePaddle和PaddleSlim框架下开发的非结构化稀疏算法，`MobileNetV1`在`ImageNet`上的稀疏化实验中，剪裁率55.19%，达到无损的表现。

本示例将演示基于不同的剪裁模式（阈值/比例）进行非结构化稀疏。默认会自动下载并使用`MNIST`数据集。当前示例目前支持`MobileNetV1`，使用其他模型可以按照下面的**训练代码示例**进行API调用。另外，为提升大稀疏度下的稀疏模型精度，我们引入了`GMP`训练策略(`Gradual Magnititude Pruning`)，使得稀疏度在训练过程中逐步增加。`GMP`训练策略在[这里](./README_GMP.md)介绍。

## 版本要求
```bash
python3.5+
paddlepaddle>=2.2.0
paddleslim>=2.2.0
```

请参照github安装[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)和[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)。

## 数据准备

本示例支持`MNIST`和`ImageNet`两种数据。默认情况下，会自动下载并使用`MNIST`数据，如果需要使用`ImageNet`数据。请按以下步骤操作：

- 根据分类模型中[ImageNet数据准备文档](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification#%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)下载数据到`PaddleSlim/demo/data/ILSVRC2012`路径下。
- 使用`train.py`和`evaluate.py`运行脚本时，指定`--data`选项为`imagenet`。

如果想要使用自定义的数据集，需要重写`../imagenet_reader.py`文件，并在`train.py`中调用实现。

## 下载预训练模型

如果使用`ImageNet`数据，建议在预训练模型的基础上进行剪裁，请从[这里](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar)下载预训练模型。

下载并解压预训练模型到当前路径：

```
wget http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar -xf MobileNetV1_pretrained.tar
```

使用`train.py`脚本时，指定`--pretrained_model`加载预训练模型，`MNIST`数据无需指定。

## 自定义稀疏化方法

默认根据参数的绝对值大小进行稀疏化，且不稀疏归一化层参数。如果开发者想更改相应的逻辑，可按照下述操作：

- 可以通过重写`paddleslim.prune.unstructured_pruner.py`中的`UnstructuredPruner.update_threshold()`来定义自己的非结构化稀疏策略（目前为剪裁掉绝对值小的parameters）。
- 可以在初始化`UnstructuredPruner`时，传入自定义的`skip_params_func`，来定义哪些参数不参与剪裁。`skip_params_func`示例代码如下(路径：`paddleslim.prune.unstructured_pruner._get_skip_params()`)。默认为所有的归一化层的参数和 `bias` 不参与剪裁。

```python
def _get_skip_params(program):
    """
    The function is used to get a set of all the skipped parameters when performing pruning.
    By default, the normalization-related ones will not be pruned.
    Developers could replace it by passing their own function when initializing the UnstructuredPruner instance.
    Args:
      - program(paddle.static.Program): the current model.
    Returns:
      - skip_params(Set<String>): a set of parameters' names.
    """
    skip_params = set()
    graph = paddleslim.core.GraphWrapper(program)
    for op in graph.ops():
        if 'norm' in op.type() and 'grad' not in op.type():
            for input in op.all_inputs():
                skip_params.add(input.name())
    for param in program.all_parameters():
        if len(param.shape) == 1:
            skip_params.add(param.name)  
    return skip_params
```

## 训练

按照阈值剪裁，GPU单卡训练：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 64 --data imagenet --lr 0.05 --pruning_mode threshold --threshold 0.01
```

按照比例剪裁，GPU单卡训练：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 64 --data imagenet --lr 0.05 --pruning_mode ratio --ratio 0.55
```

GPU多卡训练：由于静态图多卡训练方式与非结构化稀疏中的mask逻辑存在兼容性问题，会在一定程度上影响训练精度，我们建议使用[Fleet](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/training/fleet_api_howto_cn.html)方式启动稀疏化多卡训练，实测精度与单卡一致。同时，为帮助开发者将`with_data_parallel`方式配置的分布式代码转换为`Fleet`我们在[示例代码](./train.py)里面也用`"Fleet step"`清晰标注出了用代码需要做的更改
```bash
python -m paddle.distributed.launch \
          --selected_gpus="0,1,2,3" \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --lr 0.05 \
          --pruning_mode ratio \
          --ratio 0.55 \
          --is_distributed True
```

恢复训练(请替代命令中的`dir/to/the/saved/pruned/model`和`LAST_EPOCH`)：
```
CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 64 --data imagenet --lr 0.05 --pruning_mode threshold --threshold 0.01 \
                                            --checkpoint dir/to/the/saved/pruned/model --last_epoch LAST_EPOCH
```

**注意**，上述命令中的`batch_size`为单张卡上的`batch_size`。

## 推理
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --pruned_model models/ --data imagenet
```

剪裁训练代码示例：
```python
# model definition
places = paddle.static.cuda_places()
place = places[0]
exe = paddle.static.Executor(place)
model = models.__dict__[args.model]()
out = model.net(input=image, class_dim=class_dim)
cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
avg_cost = paddle.mean(x=cost)
acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

val_program = paddle.static.default_main_program().clone(for_test=True)

opt, learning_rate = create_optimizer(args, step_per_epoch)
opt.minimize(avg_cost)

#STEP1: initialize the pruner
pruner = UnstructuredPruner(paddle.static.default_main_program(), mode='threshold', threshold=0.01, place=place) # 按照阈值剪裁
# pruner = UnstructuredPruner(paddle.static.default_main_program(), mode='ratio', ratio=0.55, place=place) # 按照比例剪裁

exe.run(paddle.static.default_startup_program())
paddle.fluid.io.load_vars(exe, args.pretrained_model)

for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader):
        loss_n, acc_top1_n, acc_top5_n = exe.run(
            train_program,
            feed=data,
            fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])  
        learning_rate.step()
        #STEP2: update the pruner's threshold given the updated parameters
        pruner.step()

    if epoch % args.test_period == 0:
        #STEP3: before evaluation during training, eliminate the non-zeros generated by opt.step(), which, however, the cached masks setting to be zeros.
        pruner.update_params()
        eval(epoch)

    if epoch % args.model_period == 0:
        # STEP4: same purpose as STEP3
        pruner.update_params()
        save(epoch)
```

剪裁后测试代码示例：
```python
# intialize the model instance in static mode
# load weights
print(UnstructuredPruner.total_sparse(paddle.static.default_main_program()))
#注意，total_sparse为静态方法(static method)，可以不创建实例(instance)直接调用，方便只做测试的写法。
test()
```

更多使用参数请参照shell文件，或者通过运行以下命令查看：
```bash
python train.py --h
python evaluate.py --h
```

## 实验结果

| 模型 | 数据集 | 压缩方法 | 稀疏度 | 稀疏模型精度 | 精度变化 |
|:--:|:---:|:--:|:--:|:--:|:--:|
| MobileNetV1 | ImageNet | Baseline | - | 70.99% | - |
| MobileNetV1 | ImageNet |   ratio  | 55.19% | 70.87% | -0.12% |
| MobileNetV1 | ImageNet |   threshold  | 49.49% | 71.22% | +0.23% |
| MobileNetV1 | Imagenet | ratio, 1x1conv, GMP | 75% | 70.49% | -0.50% |
| MobileNetV1 | Imagenet | ratio, 1x1conv, GMP, 半结构化稀疏 | 75% | 68.80% | -2.19% |
| MobileNetV1 | Imagenet | ratio, 1x1conv, GMP | 80% | 70.02% | -0.97% |
| YOLO v3     |  VOC     | Baseline | - |76.24% | - |
| YOLO v3     |  VOC     |threshold | 56.50% | 77.21% | +0.97% |
| PicoDet-m-1.0 | COCO   | Baseline | - | 30.90% | - |
| PicoDet-m-1.0 | COCO   | ratio, 1x1conv, GMP | 75% | 29.40% | -1.50% |
| PP-HumanSeg-Lite | 人像分割数据集 | Baseline | - | 92.87% | - |
| PP-HumanSeg-Lite | 人像分割数据集 | ratio, 1x1conv, GMP | 75% | 92.57% | -0.30% |
| PP-HumanSeg-Lite | 人像分割数据集 | ratio, 1x1conv, GMP, 半结构化稀疏 | 75% | 92.20% (优化中) | -0.67% |

**术语说明**

Baseline: 未经压缩的稠密模型

ratio/threshold： [按照比例或者阈值稀疏](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/prune/unstructured_prune_api.rst#unstrucuturedpruner)

1x1conv： [只稀疏网络中的 1x1 卷积参数](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/prune/unstructured_prune_api.rst#unstrucuturedpruner)

GMP：[渐进稀疏算法](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/unstructured_prune/README_GMP.md)

半结构化稀疏：按照 [m=2, n=1](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/static/prune/unstructured_prune_api.rst#unstrucuturedpruner) 的方式稀疏
