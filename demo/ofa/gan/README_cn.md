# GAN压缩
基于paddle版本的 [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/abs/2003.08936)

Paddle版本：[develop](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-release)2020.07.09之后的版本。

模型压缩意义：

压缩结果：
| 模型 | 生成网络结构 | 数据集 | GMACs | 模型体积（MB）| 评价指标（fid） | 下载 |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|
|CycleGAN|resnet_9blocks(origin)|horse2zebra|56.8|11.3|64.95 |[链接]()|
|CycleGAN|mobile_resnet_9blocks(final)|horse2zebra|2.67|0.34|64.57 |[链接]()|

<p align="center">
<strong>表1-1: CycleGAN精度和计算量对比</strong>
</p>

压缩后模型参数量减少97%，计算量减少95.3%。

压缩前后模型的耗时如下表所示：

<table style="width:100%;" cellpadding="2" cellspacing="0" border="1" bordercolor="#000000">
        <tbody>
                <tr>
                        <td style="text-align:center">
                                <span style="font-size:18px;">Device</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px;">Batch Size</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px;">Model</span>
                        </td>
                        <td style="text-align:center;">
                                <span style="font-size:18px;">Latency</span>
                        </td>
                </tr>
                <tr>
                        <td rowspan=2 align=center> V100 </td>
                        <td rowspan=2 align=center> 16 </td>
                        <td rowspan=1 align=center> CycleGAN </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">117.93 ms</span>
                        </td>
                </tr>
                <tr>
                        <td rowspan=1 align=center>Compressed CycleGAN </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">33.52 ms</span>
                        </td>
                </tr>
                <tr>
                        <td rowspan=2 align=center> Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz </td>
                        <td rowspan=2 align=center> 16 </td>
                        <td rowspan=1 align=center> CycleGAN </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">56.58 s</span>
                        </td>
                </tr>
                <tr>
                        <td style="text-align:center">
                                <span style="font-size:18px;">Compressed CycleGAN</span>
                        </td>
                        <td style="text-align:center">
                                <span style="font-size:18px">9.30 s</span>
                        </td>
                </tr>
        </tbody>
</table>
<br />
<p align="center">
<strong>表1-2: 模型速度对比</strong>

压缩后模型在V100机器上相比原始模型在FP32的情况下加速252%。  
压缩后模型在Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz上相比原始模型加速508%。  
Note: 目前仅验证过CycleGAN在horse2zebra数据集中的效果，欢迎您在新的数据集或者新的模型上验证整个流程。

# CycleGAN简单介绍和压缩原理介绍
CycleGAN原理
详细讲解每个步骤原理：
....

整体压缩流程包含以下步骤：
1. 替换生成网络里的resnet block 为mobile resnnet block.
2. 裁剪掉第一步得到的生成器中的一些通道作为学生网络，第一步得到的生成器作为教师网络，蒸馏学生网络。
3. 第二步中的学生网络作为[Once-For-ALL](https://arxiv.org/abs/1908.09791)这个网络中的超网络进行训练，优化不同的子结构的效果。
4. 在第三步得到的超网络中进行搜索，得到不同子结构的FLOPs和具体评价指标。
5. （可选）微调第4步中的生成网络，这一步只对某些模型的某些数据集有效，需要具体实验得到finetune收益。
6. 导出最终模型。

## 快速开始

1. 准备cyclegan所需要的数据，数据类型:
```
├── trainA 训练数据A的目录
│   ├── img1.jpg
│   ├── ...
│  
├── trainB 训练数据B的目录
│   ├── img1.jpg
│   ├── ...
│  
├── trainA.txt 包含训练数据A名称的文件，每行代表训练数据A中的一张图片
│  
├── trainB.txt 包含训练数据B名称的文件，每行代表训练数据B中的一张图片
```

2. 开始得到压缩模型，包含步骤1~3。
```python
sh run.sh
```

3. 搜索合适的子结构。
```python
python search.py
```

4. （可选）微调第三步选出的子结构
```python
python finetune.sh
```

5. 导出最终模型。
```python
python export.py --h
```
