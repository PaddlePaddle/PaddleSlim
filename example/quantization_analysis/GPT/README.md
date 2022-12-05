# GPT量化训练敏感度分析示例


## 1. 简介
本示例将以自然语言处理生成模型GPT-3为例，介绍如何使用量化训练敏感度分析工具分析量化模型，以及提升量化训练精度。

## 2.Benchmark
| 模型  |  策略  | ACC | Inference模型 |
| :-------- |:-------- | :--------: | :--------: |
| GPT-345M | Baseline | 44.17 | [Model](https://bj.bcebos.com/v1/paddle-slim-models/GPT_345M_Baseline.tar) |
| GPT-345M | 量化训练(分析前) | 41.58 | [Model](https://bj.bcebos.com/v1/paddle-slim-models/GPT_345_QAT_wo_analysis.tar) |
| GPT-345M | 量化训练(分析后)  | 44.94 | [Model](https://bj.bcebos.com/v1/paddle-slim-models/GPT_345M_QAT_w_analysis_infer.tar) |


- ACC的指标均在基于[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)数据集，采用 ACC(accuracy) 指标评测得到

## 3. 量化分析流程
#### 3.1 准备环境
- PaddlePaddle >= 2.3 （可从[Paddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)下载安装）
- PaddleSlim develop版本
- PaddleFleetX >= 2.4

#### 3.2 准备数据集

量化敏感度分析基于验证集获得每层的敏感度，可下载和使用 [LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl) 或者 [WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) 数据集。本示例使用LAMBADA数据集来进行敏感度分析。

#### 3.3 准备预测模型
- [GPT-345M](https://bj.bcebos.com/v1/paddle-slim-models/GPT_345M_Baseline.tar) ：Base模型
- [GPT-345M](https://bj.bcebos.com/v1/paddle-slim-models/GPT_345_QAT_wo_analysis.tar) ：分析前量化训练后的模型


#### 3.4 量化敏感度分析
量化敏感度分析示例通过analysis.py脚本启动，会使用接口```paddleslim.quant.AnalysisQAT```对模型进行敏感度分析。配置config文件中模型路径、数据路径和量化相关的参数，配置完成后便可对模型进行敏感度分析。具体运行命令为：

```shell
python analysis.py --config_path=./configs/gpt_345M_analysis.yaml
```

分析完成后，会产生排序好的层敏感度（敏感度由大到小排序，敏感度越大说明约负向影响模型精度），并保存在```analysis_results/analysis.txt```中。
敏感度排序前10层分别为：```linear_31```，```linear_27```，```linear_22```，```linear_43```，```linear_83```，```linear_15```，```linear_87```，```linear_3```，```linear_38```，```linear_39```。在这十层中，其中有八层属于```TransformerDecoder```中第二个FFN层，两层属于```TransformerDecoder```中第一个FFN层，而```MultiHeadAttention```中的Linear层都相对不敏感。

```paddleslim.quant.AnalysisQAT```详解见[AnalysisQAT.md](../../../docs/zh_cn/tutorials/quant/AnalysisQAT.md)。

#### 3.5 重新量化训练

根据分析结果，重新量化训练时，去掉了```linear_31```，```linear_27```，```linear_22```，```linear_43```，```linear_83```，```linear_15```，```linear_87```七层Linear的量化，最后量化模型精度达到44.94。
