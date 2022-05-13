# 自动压缩应用 （Auto Compression Tookit）

## 简介
PaddleSlim推出全新自动压缩工具（ACT），旨在通过Source-Free的方式，自动对预测模型进行压缩，压缩后模型可直接部署应用。ACT自动压缩工具主要特性如下：
- **『更便捷』**：开发者无需了解或修改模型源码，直接使用导出的预测模型进行压缩；
- **『更智能』**：开发者简单配置即可启动压缩，ACT工具会自动优化得到最好预测模型；
- **『更丰富』**：ACT中提供了量化训练、蒸馏、结构化剪枝、非结构化剪枝、离线量化超参搜索等方法，可任意搭配使用。


## 环境准备

- 安装PaddlePaddle >= 2.3版本 （从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- 安装PaddleSlim >= 2.3 或者适当develop版本

## 快速上手

```python
# 导入依赖包
from paddleslim.auto_compression.config_helpers import load_config
from paddleslim.auto_compression import AutoCompression
# 加载配置文件
compress_config, train_config = load_slim_config("/path/to/ACT_config.yaml")
# 定义DataLoader
train_loader = reader() # DataLoader
# 开始自动压缩
ac = AutoCompression(
    model_dir="/path/to/model_dir",
    model_filename="model.pdmodel",
    params_filename="model.pdiparams",
    save_dir="output",
    strategy_config=compress_config,
    train_config=train_config,
    train_dataloader=train_loader,
    eval_callback=None)  # eval_function to verify accuracy
ac.compress()
```

**提示：**
- DataLoader传入的数据集是待压缩模型所用的数据集，DataLoader继承自`paddle.io.DataLoader`。
- 如无需验证自动压缩过程中模型的精度，`eval_callback`可不传入function，程序会自动根据损失来选择最优模型。
- 自动压缩Config中定义量化、蒸馏、剪枝等压缩算法会合并执行，压缩策略有：量化+蒸馏，剪枝+蒸馏等等。

## Benchmark

#### [目标检测](./detection)

| 模型  |  策略  | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | 预测时延<sup><small>FP32</small><sup><br><sup>(ms) |预测时延<sup><small>FP32</small><sup><br><sup>(ms) | 预测时延<sup><small>INT8</small><sup><br><sup>(ms) |  配置文件 | Inference模型  |
| :-------- |:-------- |:--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: |
| PP-YOLOE-l |  Baseline | 640*640  |  50.9   |   11.2  |   7.7ms   |  -  |  [config](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/detection/ppyoloe_crn_l_300e_coco.tar) |
| PP-YOLOE-l |  量化+蒸馏 | 640*640  |  49.5   |   - |   -   |  6.7ms  |  [config](./detection/configs/ppyoloe_l_qat_dist.yaml) | [Model](https://bj.bcebos.com/v1/paddle-slim-models/detection/ppyoloe_crn_l_300e_coco_quant.tar) |

- 测试环境：`Tesla V100 GPU`；
- 测试数据集：COCO Val2017数据集。

#### [语义分割](./semantic_segmentation)

| 模型 | 策略  | Total IoU | 耗时(ms)<br>thread=1 | 配置文件 | Inference模型  |
|:-----:|:-----:|:----------:|:---------:| :------:|:------:|
| PP-HumanSeg-Lite | Baseline |  0.9287 | 56.363 | - | [model](https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz) |
| PP-HumanSeg-Lite | 非结构化稀疏+蒸馏 |  0.9235 | 37.712 | [config](./semantic_segmentation/configs/pp_human_sparse_dis.yaml)| - |
| PP-HumanSeg-Lite | 量化+蒸馏 |  0.9284 | 49.656 | [config](./semantic_segmentation/configs/pp_human_sparse_dis.yaml) | - |

- 测试环境：`SDM710 2*A75(2.2GHz) 6*A55(1.7GHz)`；
- 测试数据集：AISegment + PP-HumanSeg14K + 内部自建数据集。

## 常见问题

#### 1.各压缩策略超参怎么设置？

-  量化：

量化参数主要设置量化比特数和量化op类型，其中量化op包含卷积层（conv2d, depthwise_conv2d）和全连接层（mul, matmul_v2）。以下为只量化卷积层的示例：
```yaml
Quantization:
    use_pact: true                                # 量化训练是否使用PACT方法
    activation_bits: 8                            # 激活量化比特数
    weight_bits: 8                                # 权重量化比特数
    activation_quantize_type: 'range_abs_max'     # 激活量化方式
    weight_quantize_type: 'channel_wise_abs_max'  # 权重量化方式
    is_full_quantize: false                       # 是否全量化
    not_quant_pattern: [skip_quant]               # 跳过量化层的name_scpoe命名(保持默认即可)
    quantize_op_types: [conv2d, depthwise_conv2d] # 量化OP列表
```

- 蒸馏：

蒸馏参数主要设置蒸馏节点（`distill_node_pair`）和教师预测模型路径。蒸馏节点需包含教师网络节点和对应的学生网络节点，其中教师网络节点名称将在程序中自动添加 “teacher_” 前缀，如下所示：
```yaml
Distillation:
    distill_lambda: 1.0
    distill_loss: l2_loss
    distill_node_pair:
    - teacher_relu_30.tmp_0
    - relu_30.tmp_0
    merge_feed: true
    teacher_model_dir: ./inference_model
    teacher_model_filename: model.pdmodel
    teacher_params_filename: model.pdiparams
```

- 非结构化稀疏参数设置如下所示，其中参数含义详见[非结构化稀疏API文档](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/pruners/unstructured_pruner.rst)：
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

#### 2.怎么选择蒸馏节点？

首先使用[Netron工具](https://netron.app/) 可视化`model.pdmodel`模型文件，选择模型中某些层输出Tensor名称，对蒸馏节点进行配置。（一般选择Backbone或网络的输出等层进行蒸馏）

<div align="center">
    <img src="../../docs/images/dis_node.png" width="600">
</div>


## 其他

如果你发现任何关于ACT自动压缩工具的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSlim/issues)给我们提issues。同时欢迎贡献更多优秀模型，共建开源生态。
