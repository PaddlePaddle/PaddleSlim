# GAN压缩
基于paddle版本的 [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/abs/2003.08936)

包含以下步骤：
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
