# 非结构化稀疏 -- 静态图剪裁（包括按照阈值和比例剪裁两种模式）示例

## 简介

在模型压缩中，常见的稀疏方式为结构化和非结构化稀疏，前者在某个特定维度（特征通道、卷积核等等）上进行稀疏化操作；后者以每一个参数为单元进行稀疏化，并不会改变参数矩阵的形状，所以更加依赖于硬件对稀疏后矩阵运算的加速能力。本目录即在PaddlePaddle和PaddleSlim框架下开发的非结构化稀疏算法，`MobileNetV1`在`ImageNet`上的稀疏化实验中，剪裁率55.19%，达到无损的表现。

同时，为了提升稀疏化训练的精度，本示例采用了GMP训练策略(Gradual Magnitude Pruning)。并且改善了`ratio`模式下训练速度慢的问题。

本示例将演示基于**比例模式**进行非结构化稀疏。默认会自动下载并使用`MNIST`数据集。当前示例目前支持`MobileNetV1`，使用其他模型可以按照下面的**训练代码示例**进行API调用。

## 版本要求
```bash
python3.5+
paddlepaddle>=2.0.0
paddleslim>=2.1.0
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
- 可以在初始化`UnstructuredPruner`时，传入自定义的`skip_params_func`，来定义哪些参数不参与剪裁。`skip_params_func`示例代码如下(路径：`paddleslim.prune.unstructured_pruner._get_skip_params()`)。默认为所有的归一化层的参数不参与剪裁。

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
    return skip_params
```

## 训练

按照比例剪裁
```bash
CUDA_VISIBLE_DEVICES=2,3 python3.7 train_parameters_easy.py --batch_size 256 --data imagenet --lr 0.005 --pruning_mode ratio --ratio 0.75 --initial_ratio 0.15 --num_epochs 108 --test_period 5 --stable_epochs 0 --pruning_epochs 54 --tunning_epochs 54 --pruning_steps 100 --step_epochs 71 88
```

关键参数的设置：

lr：假设预训练模型时，lr从0.1衰减到0.001，那么此时的lr=0.01（即log中值）。本示例中，预训练从0.1衰减到0.0001，金丝取0.005。

ratio：剪裁的最终稀疏度。实测0.75会达到比较理想的精度和加速收益平衡。

initial_ratio：剪裁的初始稀疏度。设置为0.15即可。

stable_epochs：剪枝前的稳定训练，实验看来对结果影响不大，设置为0。

pruning_epochs, tunning_epochs：两者之和为预训练时长的80%，两者等长。如果发现tunning_epoch期间，精度恢复不达预期，可以适当增加该长度。

pruning_steps：在pruning_epochs中，增加多少次ratio，使其从初始数值增加到最终数值。

step_epochs：lr piecewise decay的时间，实验看来在pruning_epochs+tunning_epochs/3，pruning_epochs+2*tunning_epochs/3 衰减两次比较好。

## 推理
```bash
CUDA_VISIBLE_DEVICES=0 python3.7 evaluate.py --pruned_model models --data imagenet
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

#STEP0: 重新定义skip_params_func，只稀疏化1x1卷积权重
def get_skip_params(program):
    skip_params = set()
    graph = GraphWrapper(program)
    for op in graph.ops():
        if 'norm' in op.type() and 'grad' not in op.type():
            for input in op.all_inputs():
                skip_params.add(input.name())
    for param in program.all_parameters():
        cond = len(param.shape) == 4 and param.shape[2] == 1 and param.shape[3] == 1
        if not cond: skip_params.add(param.name)

    return skip_params

#STEP1: initialize the pruner
pruner = UnstructuredPruner(paddle.static.default_main_program(), mode='ratio', ratio=0.0, place=place, skip_params_func=get_skip_params) # 按照比例剪裁

#STEP2: initialize the GMP instance
gmp = GMP(pruner,
          stable_iterations=0,
          pruning_iterations=54*5000, # pruning_epochs * step_per_epoch
          pruning_steps=100,
          ratio=0.75,
          initial_ratio=0.15)

exe.run(paddle.static.default_startup_program())
paddle.fluid.io.load_vars(exe, args.pretrained_model)

global_batch_id = 0
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader):
        loss_n, acc_top1_n, acc_top5_n = exe.run(
            train_program,
            feed=data,
            fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])  
        #STEP3: step the GMP hyper parameters
        gmp.step(global_batch_id) # or the global iteration id
        learning_rate.step()
        global_batch_id += 1

    if epoch % args.test_period == 0:
        eval(epoch)

    if epoch % args.model_period == 0:
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
python3.7 train.py --h
python3.7 evaluate.py --h
```

## 实验结果

| 模型 | 数据集 | 压缩方法 | 压缩率| Top-1/Top-5 Acc | lr | threshold | epoch |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileNetV1 | ImageNet | Baseline | - | 70.99%/89.68% | - | - | - |
| MobileNetV1 | ImageNet |   ratio  | -55.19% | 70.87%/89.80% (-0.12%/+0.12%) | 0.05 | - | 68 |
| MobileNetV1 | ImageNet |   threshold  | -49.49% | 71.22%/89.78% (+0.23%/+0.10%) | 0.05 | 0.01 | 93 |
| YOLO v3     |  VOC     | - | - |76.24% | - | - | - |
| YOLO v3     |  VOC     |threshold | -56.50% | 77.21%(+0.97%) | 0.001 | 0.01 |150k iterations|
