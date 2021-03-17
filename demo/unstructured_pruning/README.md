# 非结构化稀疏

## 简介

在模型压缩中，常见的稀疏方式为结构化和非结构化稀疏，前者在某个特定维度（特征通道、卷积核等等）上进行稀疏化操作；后者以每一个参数为单元进行稀疏化，所以更加依赖于硬件对稀疏后矩阵运算的加速能力。本目录即在PaddlePaddle和PaddleSlim框架下开发的非结构化稀疏算法，MobileNetV1在ImageNet上的稀疏化实验中，剪裁率41.11%，达到无损的表现，仍在持续开发和优化中。

## 版本要求
```bash
python3.5+
paddlepaddle>=2.0.0
paddleslim>=2.0.0
```

请参照github安装[paddlepaddle](https://github.com/PaddlePaddle/Paddle)和[paddleslim](https://github.com/PaddlePaddle/PaddleSlim)。

## 使用

训练前：
- 预训练模型下载，并放到某目录下，通过train.py中的--pretrained_model设置。
- 训练数据下载后，可以通过重写../imagenet_reader.py文件，并在train.py文件中调用实现。
- 确定稀疏化阈值，在当前版本下，稀疏化程度通过手动设置的阈值控制（默认为1e-5，在MobileNetV1在ImageNet上测试的经验值）。具体设置时，可以**测试**在某一阈值下，模型初始稀疏度和训练过程中稀疏度的下降快慢来调节：当发现应用阈值后，初始稀疏度过低或者训练中稀疏度下降太快，可以调低该阈值，反之调高。注意，上句话中**测试**可以通过调用UnstructurePruner.total_sparse()函数实现。

训练：
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --pretrained_model ../pretrained_model/MobileNetV1_pretrained/ --data imagenet --lr 0.05
```

推理：
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --pretrained_model models/ --phase test --data imagenet
```

更多使用参数请参照如下，请按照实际数据集和GPU资源进行调整：
```bash
usage: train.py [-h] [--batch_size BATCH_SIZE] [--use_gpu USE_GPU]
                [--model MODEL] [--pretrained_model PRETRAINED_MODEL]
                [--lr LR] [--lr_strategy LR_STRATEGY] [--l2_decay L2_DECAY]
                [--momentum_rate MOMENTUM_RATE] [--threshold THRESHOLD]
                [--num_epochs NUM_EPOCHS]
                [--step_epochs STEP_EPOCHS [STEP_EPOCHS ...]] [--data DATA]
                [--log_period LOG_PERIOD] [--phase PHASE]
                [--test_period TEST_PERIOD] [--model_path MODEL_PATH]
                [--model_period MODEL_PERIOD] [--resume_epoch RESUME_EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Minibatch size. Default: 512.
  --use_gpu USE_GPU     Whether to use GPU or not. Default: True.
  --model MODEL         The target model. Default: MobileNet.
  --pretrained_model PRETRAINED_MODEL
                        Whether to use pretrained model. Default:
                        ../pretrained_model/MobileNetV1_pretrained.
  --lr LR               The learning rate used to fine-tune pruned model.
                        Default: 0.1.
  --lr_strategy LR_STRATEGY
                        The learning rate decay strategy. Default:
                        piecewise_decay.
  --l2_decay L2_DECAY   The l2_decay parameter. Default: 3e-05.
  --momentum_rate MOMENTUM_RATE
                        The value of momentum_rate. Default: 0.9.
  --threshold THRESHOLD
                        The threshold to set zeros, the abs(weights) lower
                        than which will be zeros. Default: 1e-05.
  --num_epochs NUM_EPOCHS
                        The number of total epochs. Default: 120.
  --step_epochs STEP_EPOCHS [STEP_EPOCHS ...]
                        piecewise decay step
  --data DATA           Which data to use. 'mnist' or 'imagenet'. Default:
                        mnist.
  --log_period LOG_PERIOD
                        Log period in batches. Default: 100.
  --phase PHASE         Whether to train or test the pruned model. Default:
                        train.
  --test_period TEST_PERIOD
                        Test period in epoches. Default: 10.
  --model_path MODEL_PATH
                        The path to save model. Default: ./models.
  --model_period MODEL_PERIOD
                        The period to save model in epochs. Default: 10.
  --resume_epoch RESUME_EPOCH
                        The epoch to resume training. Default: 0.
```

## 实验结果

| 模型 | 数据集 | 压缩方法 | 压缩率| Top-1/Top-5 Acc | lr | threshold | epoch |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileNetV1 | ImageNet | Baseline | - | 70.99%/89.68% | - | - | - |
| MobileNetV1 | ImageNet | Global Pruning | -41.11% | 71.09%/89.73% (+0.10%/+0.05%) | 0.05 | 0.00001 | 40 |

## TODO

- [ ] 用固定比例（例如，剪裁掉30%的weights）替代手动寻找threshold的方式，简化操作流程。开发并实现。
- [ ] 完善实验，验证该方法可以达到的最大稀疏度（接近精度无损的表现下）。
