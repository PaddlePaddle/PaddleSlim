# Distillation example: Chinese lexical analysis
We demonstrated how to use the Pantheon framework for online distillation of the Chinese lexical analysis model with sample dataset. The effect of large-scale online distillation is shown below:
| model | Precision | Recall | F1-score|
| ------ | ------ | ------ | ------ |
| BiGRU | 89.2 | 89.4 | 89.3 |
| BERT fine-tuned | 90.2 | 90.4 | 90.3 |
| ERNIE fine-tuned | 91.7 | 91.7 | 91.7 |
| DistillBiGRU | 90.20  | 90.52 | 90.36 |

BiGRU is to train a BiGRU based LAC model from scratch; BERT fine-tuned is to fine-tune LAC task on BERT base model; ERNIE fine-tuned is to fine-tune LAC task on BERT base model; DistillBiGRU is trained through large-scale online distillation with ERNIE fine-tuned as teacher model.

## Introduction

Lexical Analysis of Chinese, or LAC for short, is a lexical analysis model that completes the tasks of Chinese word segmentation, part-of-speech tagging, and named entity recognition in a single model. We conduct an overall evaluation of word segmentation, part-of-speech tagging, and named entity recognition on a self-built dataset. We use the finetuned [ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE) model as the Teacher model and GRU as the Student model, which are needed by the Pantheon framework for online distillation.

#### 1. Download the training data set

Download the data set file, and after decompression, a `./data/` folder will be created.
```bash
python downloads.py dataset
```

#### 2. Download the Teacher model

```bash
# download ERNIE finetuned model
python downloads.py finetuned
python downloads.py conf
```

### 3. Distilling Student model
```bash
# start teacher service
bash run_teacher.sh

# start student service
bash run_student.sh
```

> If you want to learn more about LAC, you can refer to this repo: https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis