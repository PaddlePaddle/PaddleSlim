# 蒸馏样例：中文词法分析
我们在样例数据集上，对中文词法分析模型，演示了如何使用Pantheon框架进行在线蒸馏。大规模在线蒸馏的效果如下图所示：

| 模型 | 精度 | 召回率 | F1值|
| ------ | ------ | ------ | ------ |
| BiGRU | 89.2 | 89.4 | 89.3 |
| BERT fine-tuned | 90.2 | 90.4 | 90.3 |
| ERNIE fine-tuned | 91.7 | 91.7 | 91.7 |
| DistillBiGRU | 90.20  | 90.52 | 90.36 |

BiGRU 是使用双向GRU网络从头训练LAC任务；BERT fine-tuned 是在BERT base模型上微调LAC任务；ERNIE fine-tuned 是在ERNIE base模型上微调LAC任务；DistillBiGRU 是使用ERNIE fine-tuned模型作为teacher模型，通过大规模蒸馏训练LAC任务。

## 简介

Lexical Analysis of Chinese，简称 LAC，是一个联合的词法分析模型，在单个模型中完成中文分词、词性标注、专名识别任务。我们在自建的数据集上对分词、词性标注、专名识别进行整体的评估效果。我们使用经过finetune的 [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) 模型作为Teacher模型，使用GRU作为Student模型，使用Pantheon框架进行在线蒸馏。

#### 1. 下载训练数据集

下载数据集文件，解压后会生成 `./data/` 文件夹
```bash
python downloads.py dataset
```

#### 2. 下载Teacher模型

```bash
# download ERNIE finetuned model
python downloads.py finetuned
python downloads.py conf
```

### 3. 蒸馏Student模型
```bash
# start teacher service
bash run_teacher.sh

# start student service
bash run_student.sh
```

> 如果你想详细了解LAC的原理可以参照相关repo: https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis
