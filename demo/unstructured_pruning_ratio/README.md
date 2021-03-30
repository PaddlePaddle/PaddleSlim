# 非结构化稀疏

## 简介

在模型压缩中，常见的稀疏方式为结构化和非结构化稀疏，前者在某个特定维度（特征通道、卷积核等等）上进行稀疏化操作；后者以每一个参数为单元进行稀疏化，所以更加依赖于硬件对稀疏后矩阵运算的加速能力。本目录即在PaddlePaddle和PaddleSlim框架下开发的非结构化稀疏算法，MobileNetV1在ImageNet上的稀疏化实验中，剪裁率55.19%，达到无损的表现。

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

训练：
```bash
python3 train.py --phase train --data imagenet --lr 0.005
```

推理：
```bash
python3 train.py --pretrained_model models/ --data imagenet --phase test
```

更多使用参数请参照如下，请按照实际数据集和GPU资源进行调整：
```bash
usage: train.py [-h] [--batch_size BATCH_SIZE] [--use_gpu USE_GPU]
                [--model MODEL] [--pretrained_model PRETRAINED_MODEL]
                [--lr LR] [--lr_strategy LR_STRATEGY] [--l2_decay L2_DECAY]
                [--momentum_rate MOMENTUM_RATE] [--num_epochs NUM_EPOCHS]
                [--step_epochs STEP_EPOCHS [STEP_EPOCHS ...]]
                [--config_file CONFIG_FILE] [--data DATA]
                [--log_period LOG_PERIOD] [--test_period TEST_PERIOD]
                [--model_path MODEL_PATH] [--model_period MODEL_PERIOD]
                [--min_ratio MIN_RATIO] [--max_ratio MAX_RATIO]
                [--resume_epoch RESUME_EPOCH] [--pruned_ratio PRUNED_RATIO]
                [--criterion CRITERION] [--save_inference SAVE_INFERENCE]
                [--phase PHASE]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Minibatch size. Default: 256.
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
  --num_epochs NUM_EPOCHS
                        The number of total epochs. Default: 120.
  --step_epochs STEP_EPOCHS [STEP_EPOCHS ...]
                        piecewise decay step
  --config_file CONFIG_FILE
                        The config file for compression with yaml format.
                        Default: None.
  --data DATA           Which data to use. 'mnist' or 'imagenet' Default:
                        mnist.
  --log_period LOG_PERIOD
                        Log period in batches. Default: 100.
  --test_period TEST_PERIOD
                        Test period in epoches. Default: 2.
  --model_path MODEL_PATH
                        The path to save model. Default: ./models.
  --model_period MODEL_PERIOD
                        The period of model saving. Default: 10.
  --min_ratio MIN_RATIO
                        The starting pruning ratio corresponding to the
                        starting epoch. Default: 0.4.
  --max_ratio MAX_RATIO
                        The maximum pruning ratio. Default: 0.6.
  --resume_epoch RESUME_EPOCH
                        The starting epoch. Default: 0.
  --pruned_ratio PRUNED_RATIO
                        The ratios to be pruned. Default: None.
  --criterion CRITERION
                        The prune criterion to be used, support l1_norm and
                        batch_norm_scale. Default: l1_norm.
  --save_inference SAVE_INFERENCE
                        Whether to save inference model. Default: False.
  --phase PHASE         Whether to train or test the pruned model. Default:
                        train.
```

## 实验结果

| 模型 | 数据集 | 压缩方法 | 压缩率| Top-1/Top-5 Acc | lr | threshold | epoch |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileNetV1 | ImageNet | Baseline | - | 70.99%/89.68% | - | - | - |
| MobileNetV1 | ImageNet | Global Pruning | -55.19% | 70.87%/89.80% (-0.12%/+0.12%) | 0.005 | - | 68 |

## TODO

- [x] 用固定比例（例如，剪裁掉30%的weights）替代手动寻找threshold的方式，简化操作流程。开发并实现。
- [x] 完善实验，验证该方法可以达到的最大稀疏度（接近精度无损的表现下）。
- [ ] 阈值剪裁与比例剪裁功能合并。
