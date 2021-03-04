# OFA详细教程

&emsp;&emsp;[Once-For-All(以下简称OFA)](https://arxiv.org/abs/1908.09791)主要的目的是训练一个超网络，根据不同的硬件从超网络中选择满足时延要求和精度要求的小模型。可以基于已有的预训练模型进行压缩也是OFA一个很大的优势。  
&emsp;&emsp;为了防止子网络之间互相干扰，本论文提出了一种Progressive Shrinking(PS)的模式进行超网络训练，逐步从大型子网络到小型子网络进行训练。首先是从最大的子网络开始训练，例如：超网络包含可变的卷积核大小 kernel_size = {3, 5, 7}，可变的网络结构深度 depth = {2, 3, 4} 和可变的网络的宽度 expand_ratio = {2, 4, 6}，则训练卷积核为7、深度为4，宽度为6的网络。之后逐步将其添加到搜索空间中来逐步调整网络以支持较小的子网络。具体来说，在训练了最大的网络之后，我们首先支持可变卷积核大小，可以在{3，5，7}中进行选择，而深度和宽度则保持最大值。然后，我们依次支持可变深度和可变宽度。

## 使用方法

OFA的基本流程分为以下步骤：
1. 定义超网络
2. 训练配置
3. 蒸馏配置
4. 传入模型和相应配置

### 1. 定义超网络
   这里的超网络指的是用[动态OP](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/ofa/ofa_layer_api.html)组网的网络。
   PaddleSlim提供了三种获得超网络的方式，具体可以参考[超网络转换](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/ofa/convert_supernet_api.html)。

```python
  import paddle
  from paddle.vision.models import mobilenet_v1
  from paddleslim.nas.ofa.convert_super import Convert, supernet

  model = mobilenet_v1()
  sp_net_config = supernet(kernel_size=(3, 5, 7), expand_ratio=[1, 2, 4])
  sp_model = Convert(sp_net_config).convert(model)
```

### 2. 训练配置
   训练配置默认根据论文中PS的训练模式进行配置，可进行配置的参数和含义可以参考: [RunConfig](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/ofa/ofa_api.html)

```python
  from paddleslim.nas.ofa import RunConfig
  default_run_config = {
      'train_batch_size': 256,
      'n_epochs': [[1], [2, 3], [4, 5]],
      'init_learning_rate': [[0.001], [0.003, 0.001], [0.003, 0.001]],
      'dynamic_batch_size': [1, 1, 1],
      'total_images': 1281167,
      'elastic_depth': (2, 5, 8)
  }
  run_config = RunConfig(**default_run_config)
```

### 3. 蒸馏配置
  为OFA训练过程添加蒸馏配置，可进行配置的参数和含义可以参考: [DistillConfig](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/ofa/ofa_api.html#distillconfig)

```python
  from paddle.vision.models import mobilenet_v1
  from paddleslim.nas.ofa import DistillConfig
  teacher_model = mobilenet_v1()

  default_distill_config = {
      'teacher_model': teacher_model
  }
  distill_config = DistillConfig(**default_distill_config)
```

### 4. 传入模型和相应配置
  用OFA封装模型、训练配置和蒸馏配置。配置完模型和正常模型训练流程相同。如果添加了蒸馏，则OFA封装后的模型会比原始模型多返回一组教师网络的输出。
```python
  from paddleslim.nas.ofa import OFA

  ofa_model = OFA(model, run_config=run_config, distill_config=distill_config)
```

## 实验效果

目前我们进在BERT-base、TinyBERT和TinyERNIE上进行了压缩实验，其他CV任务的压缩效果之后会进行补充。BERT和TinyBERT的压缩结果如下表所示。

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<strong>表1: BERT-base上GLUE数据集精度对比</strong>
| Task  | Metric                       | BERT-base         | Result with PaddleSlim |
|:-----:|:----------------------------:|:-----------------:|:----------------------:|
| SST-2 | Accuracy                     |      0.93005      |     [0.931193]()       |
| QNLI  | Accuracy                     |      0.91781      |     [0.920740]()       |
| CoLA  | Mattehew's corr              |      0.59557      |     [0.601244]()       |
| MRPC  | F1/Accuracy                  |  0.91667/0.88235  |  [0.91740/0.88480]()   |
| STS-B | Person/Spearman corr         |  0.88847/0.88350  |  [0.89271/0.88958]()   |
| QQP   | Accuracy/F1                  |  0.90581/0.87347  |  [0.90994/0.87947]()   |
| MNLI  | Matched acc/MisMatched acc   |  0.84422/0.84825  |  [0.84687/0.85242]()   |
| RTE   | Accuracy                     |      0.711191     |     [0.718412]()       |


&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<strong>表2: TinyBERT上GLUE数据集精度对比</strong>
| Task  | Metric                       | TinyBERT(L=4, D=312) |     Result with OFA    |
|:-----:|:----------------------------:|:--------------------:|:----------------------:|
| SST-2 | Accuracy                     |     [0.9234]()       |      [0.9220]()        |
| QNLI  | Accuracy                     |     [0.8746]()       |      [0.8720]()        |
| CoLA  | Mattehew's corr              |     [0.4961]()       |      [0.5048]()        |
| MRPC  | F1/Accuracy                  |  [0.8998/0.8554]()   |   [0.9003/0.8578]()    |
| STS-B | Person/Spearman corr         |  [0.8635/0.8631]()   |   [0.8717/0.8706]()    |
| QQP   | Accuracy/F1                  |  [0.9047/0.8751]()   |   [0.9034/0.8733]()    |
| MNLI  | Matched acc/MisMatched acc   |  [0.8256/0.8294]()   |   [0.8211/0.8261]()    |
| RTE   | Accuracy                     |     [0.6534]()       |      [0.6787]()        |
