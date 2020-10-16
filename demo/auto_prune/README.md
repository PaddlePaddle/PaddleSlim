该示例介绍如何使用自动裁剪。
该示例需要使用IMAGENET数据，以及预训练模型。支持以下模型：

- MobileNetV1
- MobileNetV2
- ResNet50

## 1. 接口介绍

该示例涉及以下接口：

- [paddleslim.prune.AutoPruner])
- [paddleslim.prune.Pruner])

## 2. 运行示例


提供两种自动裁剪模式，直接以裁剪目标进行一次自动裁剪，和多次迭代的方式进行裁剪。

###2.1一次裁剪

在路径`PaddleSlim/demo/auto_prune`下执行以下代码运行示例：

```
export CUDA_VISIBLE_DEVICES=0
python train.py --model "MobileNet"
从log中获取搜索的最佳裁剪率列表，加入到train_finetune.py的ratiolist中，如下命令finetune得到最终结果
python train_finetune.py --model "MobileNet" --lr 0.1 --num_epochs 120 --step_epochs 30 60 90

```

通过`python train.py --help`查看更多选项。


###2.2多次迭代裁剪

在路径`PaddleSlim/demo/auto_prune`下执行以下代码运行示例：

```
export CUDA_VISIBLE_DEVICES=0
python train_iterator.py --model "MobileNet"
从log中获取本次迭代搜索的最佳裁剪率列表，加入到train_finetune.py的ratiolist中,如下命令开始finetune本次搜索到的结果
python train_finetune.py --model "MobileNet"
将第一次迭代的最佳裁剪率列表，加入到train_iterator.py 的ratiolist中,如下命令进行第二次迭代
python train_iterator.py --model "MobileNet" --pretrained_model "checkpoint/Mobilenet/19"
finetune第二次迭代搜索结果，并继续重复迭代，直到获得目标裁剪率的结果
...
如下命令finetune最终结果
python train_finetune.py --model "MobileNet" --pretrained_model "checkpoint/Mobilenet/19"  --num_epochs 70 --step_epochs 10 40
```


## 3. 注意

### 3.1 一次裁剪

在`paddleslim.prune.AutoPruner`接口的参数中，pruned_flops表示期望的最低flops剪切率。


### 3.2 多次迭代裁剪

单次迭代的裁剪目标，建议不高于10%。
在load前次迭代结果时，需要删除checkpoint下learning_rate、@LR_DECAY_COUNTER@等文件，避免继承之前的learning_rate，影响finetune效果。
