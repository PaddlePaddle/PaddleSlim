[English](README_en.md) | 简体中文

# SlimOCR模型库


## 模型

PaddleSlim对[PaddleOCR]()发布的模型进行了压缩，产出了如下一系列小模型：


<table>
<thead>
  <tr>
    <th>序号</th>
    <th>任务</th>
    <th>模型</th>
    <th>压缩策略<sup><a href="#quant">[3]</a><a href="#prune">[4]</a><sup></th>
    <th>精度(自建中文数据集)</th>
    <th>耗时<sup><a href="#latency">[1]</a></sup>(ms)</th>
    <th>整体耗时<sup><a href="#rec">[2]</a></sup>(ms)</th>
    <th>加速比</th>
    <th>整体模型大小(M)</th>
    <th>压缩比例</th>
    <th>下载链接</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">0</td>
    <td>检测</td>
    <td>MobileNetV3_DB</td>
    <td>无</td>
    <td>61.7</td>
    <td>224</td>
    <td rowspan="2">375</td>
    <td rowspan="2">-</td>
    <td rowspan="2">8.6</td>
    <td rowspan="2">-</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>MobileNetV3_CRNN</td>
    <td>无</td>
    <td>62.0</td>
    <td>9.52</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">1</td>
    <td>检测</td>
    <td>SlimTextDet</td>
    <td>PACT量化训练</td>
    <td>62.1</td>
    <td>195</td>
    <td rowspan="2">348</td>
    <td rowspan="2">8%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">2</td>
    <td>检测</td>
    <td>SlimTextDet_quat_pruning</td>
    <td>剪裁+PACT量化训练</td>
    <td>60.86</td>
    <td>142</td>
    <td rowspan="2">288</td>
    <td rowspan="2">30%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">3</td>
    <td>检测</td>
    <td>SlimTextDet_pruning</td>
    <td>剪裁</td>
    <td>61.57</td>
    <td>138</td>
    <td rowspan="2">295</td>
    <td rowspan="2">27%</td>
    <td rowspan="2">2.9</td>
    <td rowspan="2">66.28%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
</tbody>
</table>


**注意**:

-   <a name="latency">[1]</a> 耗时评测环境为：骁龙855芯片+PaddleLite。
-   <a name="rec">[2]</a> 整体耗时不等于检测耗时加识别耗时的原因是：识别模型的耗时为单个检测框的耗时，一张图片可能会有多个检测框。
-   <a name="quant">[3]</a> 参考下面关于[OCR量化的说明](#OCR量化说明)。
-   <a name="prune">[4]</a> 参考下面关于[OCR剪裁的说明](#OCR剪裁说明)。


## OCR量化说明

对于OCR模型，普通的量化训练精度损失较大，并且训练不稳定。所以我们选择PACT方法进行量化

### 文本检测模型

MobileNetV3_DB是一个全卷积模型，我们可以对整个模型进行量化。

整个量化训练的轮数与全精度模型的训练轮数一致，量化的配置如下所示：

```python
    quant_config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'weight_bits': 8,
        'activation_bits': 8,
        'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
        'dtype': 'int8',
        'window_size': 10000,
        'moving_rate': 0.9,
    }
```

对于PACT参数，我们沿用了论文中的方法，截断阈值$\alpha$的学习率与原模型其他参数保持一致。另外，对其增加一个系数为0.0001的L2正则化，使用`AdamOptimizer`对其进行优化，确保其能快速收敛。

### 文本识别模型

MobileNetV3_CRNN模型包含一个LSTM组件，因为暂时不支持对LSTM进行量化，我们暂时跳过这一部分。

通过[scope_guard API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/executor_cn/scope_guard_cn.html#scope-guard)将LSTM切换到新的作用域`skip_quant`，量化配置中通过`not_quant_pattern`设置不对这一部分进行量化，具体量化配置如下:
```python
    quant_config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'weight_bits': 8,
        'activation_bits': 8,
        'not_quant_pattern': ['skip_quant'],
        'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
        'dtype': 'int8',
        'window_size': 10000,
        'moving_rate': 0.9,
    }
```

同样地，量化训练的轮数与全精度模型的训练轮数一致，PACT阈值$\alpha$的学习率与原模型其他参数保持一致。我们发现，对$\alpha$使用与原模型其他参数一样的L2正则化系数，量化训练就可以很好地收敛。关于优化器，使用`AdamOptimizer`对其进行优化，确保其能快速收敛。


更多量化教程请参考[OCR模型量化压缩教程](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/quantization/README.md)


## OCR剪裁说明

### 敏感度分析
  在对OCR文字检测模型进行裁剪敏感度分析时，分析对象为除depthwise convolution外的所有普通卷积层，裁剪的criterion被设置为'geometry_median'，pruned_ratios推荐设置为[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]。

### 裁剪与finetune
  裁剪时通过之前的敏感度分析文件决定每个网络层的裁剪比例。在具体实现时，为了尽可能多的保留从图像中提取的低阶特征，我们跳过了backbone中靠近输入的4个卷积层。同样，为了减少由于裁剪导致的模型性能损失，我们通过之前敏感度分析所获得敏感度表，挑选出了一些冗余较少，对裁剪较为敏感[网络层](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/prune/pruning_and_finetune.py#L41)，并在之后的裁剪过程中选择避开这些网络层。裁剪过后finetune的过程沿用OCR检测模型原始的训练策略。


更多OCR剪裁教程请参考[OCR模剪裁压缩教程](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/prune/README.md)
