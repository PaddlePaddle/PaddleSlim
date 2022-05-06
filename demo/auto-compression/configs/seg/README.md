# 使用预测模型进行自动压缩示例

本示例将介绍如何使用PaddleSeg中预测模型进行自动压缩训练。

## 自动压缩训练流程

### 1. 准备数据集

参考[PaddleSeg数据准备文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.5/docs/data/marker/marker_cn.md)

### 2. 准备待压缩模型

PaddleSeg是基于飞桨PaddlePaddle开发的端到端图像分割开发套件，涵盖了高精度和轻量级等不同方向的大量高质量分割模型。
#### 2.1 下载代码
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```
#### 2.2 准备预训练模型

以BiseNetV2模型为例，可执行如下命令或者点击[链接](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams)下载模型预训练权重。
```
mkdir bisenet && cd bisenet
wget https://paddleseg.bj.bcebos.com/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams
cd ..
```
#### 2.3 导出预测模型

在PaddleSeg目录下执行如下命令，则预测模型会保存在output文件夹。
```shell
# 设置1张可用的卡
export CUDA_VISIBLE_DEVICES=0
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
       --model_path bisenet/model.pdparams \
       --save_dir output
       --input_shape 1 3 512 1024
```

### 3. 多策略融合压缩

每一个小章节代表一种多策略融合压缩方式。

### 3.1 进行蒸馏稀疏压缩
自动压缩训练需要准备 config 文件、数据集 dataloader 以及测试函数（``eval_function``）。
#### 3.1.1 配置config

使用自动压缩进行蒸馏和非结构化稀疏的联合训练，首先要配置config文件，包含蒸馏、稀疏和训练三部分参数。

- 蒸馏参数

蒸馏参数主要设置蒸馏节点（``distill_node_pair``）和教师网络测预测模型路径。蒸馏节点需包含教师网络节点和对应的学生网络节点，其中教师网络节点名称将在程序中自动添加“teacher_”前缀，如下所示。
```yaml
Distillation:
  distill_lambda: 1.0
  distill_loss: l2_loss
  distill_node_pair:
  - teacher_relu_30.tmp_0
  - relu_30.tmp_0
  merge_feed: true
  teacher_model_dir: ./infermodel
  teacher_model_filename: model.pdmodel
  teacher_params_filename: model.pdiparams
```
- 稀疏参数

稀疏参数设置如下所示，其中参数含义详见[非结构化稀疏API文档](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/pruners/unstructured_pruner.rst)。
```yaml
UnstructurePrune:
  prune_strategy: gmp
  prune_mode: ratio
  pruned_ratio: 0.75
  gmp_config:
    stable_iterations: 0
    pruning_iterations: 4500
    tunning_iterations: 4500
    resume_iteration: -1
    pruning_steps: 100
    initial_ratio: 0.15
  prune_params_type: conv1x1_only
  local_sparsity: True
```

- 训练参数

训练参数主要设置学习率、训练次数（epochs）和优化器等。
```yaml
TrainConfig:
  epochs: 14
  eval_iter: 400
  learning_rate: 5.0e-03
  optimizer: SGD
  optim_args:
    weight_decay: 0.0005
```
#### 3.1.2 准备dataloader和测试函数
准备好数据集后，需将训练数据的封装成dict类型传入自动压缩接口，可参考以下函数进行封装。测试函数用于测试模型精度，需在静态图模式下实现。
```python
def reader_wrapper(reader):
    def gen():
        for i, data in enumerate(reader()):
            imgs = np.array(data[0])
            yield {"x": imgs}
    return gen
```
> 注：该dict类型的key值要和保存预测模型时的输入名称保持一致。

#### 3.1.3 开启训练

将数据集dataloader和测试函数传入接口``paddleslim.auto_compression.AutoCompression``，对模型进行稀疏训练。运行指令如下：
```shell
python demo_seg.py \
    --model_dir='infermodel' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_model' \
    --devices='gpu' \
    --config_path='configs/seg/humanseg_sparse_dis.yaml'
```

### 3.2 进行蒸馏量化压缩
#### 3.2.1 配置config
使用自动压缩进行量化训练，首先要配置config文件，包含蒸馏、量化和训练三部分参数。其中蒸馏和训练参数与稀疏训练类似，下面主要介绍量化参数的设置。
- 量化参数

量化参数主要设置量化比特数和量化op类型，其中量化op包含卷积层（conv2d, depthwise_conv2d）和全连接层（matmul）。以下为只量化卷积层的示例：
```yaml
Quantization:
  activation_bits: 8
  weight_bits: 8
  is_full_quantize: false
  not_quant_pattern:
  - skip_quant
  quantize_op_types:
  - conv2d
  - depthwise_conv2d
```
#### 3.2.2 开启训练
将数据集dataloader和测试函数（``eval_function``）传入接口``paddleslim.auto_compression.AutoCompression``，对模型进行量化训练。运行指令如下：
```
python demo_seg.py \
    --model_dir='infermodel' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_model' \
    --devices='gpu' \
    --config_path='configs/seg/humanseg_quant_dis.yaml'
```
### 4. 压缩结果
我们使用PP-HumanSeg模型进行了自动压缩实验，并在SD710上使用单线程测速，压缩结果和测速结果如下：

| 压缩方式  | Total IoU | 耗时(ms)<br>thread=1 | 加速比 |
|:-----:|:----------:|:---------:| :------:|
| Baseline |  0.9287 | 56.363 | - |
| 稀疏 |  0.9235 | 37.712 | 49.456% |
| 量化 |  0.9284 | 49.656 | 13.506% |
