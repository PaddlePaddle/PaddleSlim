# 结构化剪枝敏感度分析

本示例将以自动压缩示例中MobileNetV1为例，介绍如何快速修改示例代码，进行结构化剪枝敏感度分析工具分析模型参数敏感度，从而设置合适的剪枝比例和要剪枝的参数，在保证剪枝后模型精度的前提下进行最大比例的模型剪枝。
图像分类除MobileNetV1模型外其他模型的结构化剪枝敏感度分析可以直接使用 [run.py](./run.py) 脚本，替换传入的 config_path 文件为其他模型的任一压缩yaml文件，即可对其他图像分类模型进行敏感度分析。

## 计算通道剪枝敏感度

以下为示例代码每一步的含义，如果您是ACT（自动压缩工具）的用户，加粗文字表示如何把一个自动压缩示例改为一个敏感度分析示例。

### 1. 引入依赖

引入一些需要的依赖，可以直接复用以下代码，如果您需要对其他场景下模型进行敏感度分析，需要把其他场景文件下中 ``run.py`` 文件中独有的依赖也导入进来。**或者把最后一个依赖放入自动压缩示例代码中。**

```python
import os
import sys
import argparse
import pickle
import functools
from functools import partial
import math
from tqdm import tqdm

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddleslim
from imagenet_reader import ImageNetDataset
from paddleslim.common import load_config as load_slim_config
from paddleslim.auto_compression.analysis import analysis_prune
```

### 2. 定义可传入参数

定义一些可以通过指令传入的参数。**此段代码无论您想对任何场景的模型进行分析都无需修改，复制过去替换原本的指令即可**

```python
def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--analysis_file',
        type=str,
        default='sensitivity_0.data',
        help="directory to save compressed model.")
    parser.add_argument(
        '--pruned_ratios',
        nargs='+',
        type=float,
        default=[0.1, 0.2, 0.3, 0.4],
        help="The ratios to be pruned when compute sensitivity.")
    parser.add_argument(
        '--target_loss',
        type=float,
        default=0.2,
        help="use the target loss to get prune ratio of each parameter")

    return parser


```

### 3. 定义eval_function

需要定义完整的测试流程，可以直接使用对应场景文件夹下 ``run.py`` 文件中的测试流程即可，**把自动压缩示例代码中测试回调函数中下面这一行代码:**

```python
def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
```
**修改成：**
```python
def eval_function(compiled_test_program, exe, test_feed_names, test_fetch_list):
```

最终的测试过程代码如下:
```python
def eval_reader(data_dir, batch_size, crop_size, resize_size, place=None):
    val_reader = ImageNetDataset(
        mode='val',
        data_dir=data_dir,
        crop_size=crop_size,
        resize_size=resize_size)
    val_loader = DataLoader(
        val_reader,
        places=[place] if place is not None else None,
        batch_size=global_config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=0)
    return val_loader


def eval_function(compiled_test_program, exe, test_feed_names, test_fetch_list):
    val_loader = eval_reader(
        global_config['data_dir'],
        batch_size=global_config['batch_size'],
        crop_size=img_size,
        resize_size=resize_size)

    results = []
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for batch_id, (image, label) in enumerate(val_loader):
            # top1_acc, top5_acc
            if len(test_feed_names) == 1:
                image = np.array(image)
                label = np.array(label).astype('int64')
                pred = exe.run(compiled_test_program,
                               feed={test_feed_names[0]: image},
                               fetch_list=test_fetch_list)
                pred = np.array(pred[0])
                label = np.array(label)
                sort_array = pred.argsort(axis=1)
                top_1_pred = sort_array[:, -1:][:, ::-1]
                top_1 = np.mean(label == top_1_pred)
                top_5_pred = sort_array[:, -5:][:, ::-1]
                acc_num = 0
                for i in range(len(label)):
                    if label[i][0] in top_5_pred[i]:
                        acc_num += 1
                top_5 = float(acc_num) / len(label)
                results.append([top_1, top_5])
            else:
                # eval "eval model", which inputs are image and label, output is top1 and top5 accuracy
                image = np.array(image)
                label = np.array(label).astype('int64')
                result = exe.run(compiled_test_program,
                                 feed={
                                     test_feed_names[0]: image,
                                     test_feed_names[1]: label
                                 },
                                 fetch_list=test_fetch_list)
                result = [np.mean(r) for r in result]
                results.append(result)
            t.update()
    result = np.mean(np.array(results), axis=0)
    return result[0]
```

### 4. 加载配置文件
加载配置文件，获得文件中数据读取部分的相关配置。**使用原始的自动压缩示例代码中的即可**
```python
global global_config
all_config = load_slim_config(args.config_path)

assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
global_config = all_config["Global"]

global img_size, resize_size
img_size = global_config['img_size'] if 'img_size' in global_config else 224
resize_size = global_config[
    'resize_size'] if 'resize_size' in global_config else 256
```

### 4. 进行敏感度分析

传入测试回调函数，配置（主要包括模型位置和模型名称等信息），分析文件保存的位置，要分析的裁剪比例和可以接受的精度目标损失。如果不传入可以接受的精度目标损失，则只返回敏感度分析情况。**把自动压缩代码中调用AutoCompression 和 ac.compress 的代码替换成以下代码即可**

```python
analysis_prune(eval_function, global_config['model_dir'], global_config['model_filename'], global_config['params_filename'], args.analysis_file,
               args.pruned_ratios, args.target_loss)
```
