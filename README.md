# PaddleSlim

中文 | [English](README.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)]()
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

PaddleSlim是一个模型压缩工具库，包含模型剪裁、定点量化、知识蒸馏、超参搜索和模型结构搜索等一系列模型压缩策略。

对于业务用户，PaddleSlim提供完整的模型压缩解决方案，可用于图像分类、检测、分割等各种类型的视觉场景。
同时也在持续探索NLP领域模型的压缩方案。另外，PaddleSlim提供且在不断完善各种压缩策略在经典开源任务的benchmark,
以便业务用户参考。

对于模型压缩算法研究者或开发者，PaddleSlim提供各种压缩策略的底层辅助接口，方便用户复现、调研和使用最新论文方法。
PaddleSlim会从底层能力、技术咨询合作和业务场景等角度支持开发者进行模型压缩策略相关的创新工作。


## 功能

<table style="width:100%;" cellpadding="2" cellspacing="0" border="1" bordercolor="#000000">
	<tbody>
		<tr>
			<td style="text-align:center;">
				<span style="font-size:18px;">功能模块</span>
			</td>
			<td style="text-align:center;">
				<span style="font-size:18px;">算法</span>
			</td>
			<td style="text-align:center;">
				<span style="font-size:18px;">教程</span><span style="font-size:18px;">与文档</span>
			</td>
		</tr>
		<tr>
			<td style="text-align:center;">
				<span style="font-size:12px;">剪裁</span><span style="font-size:12px;"></span><br />
			</td>
			<td>
				<p>
					<br />
				</p>
				<ul>
					<li>
						卷积通道剪裁 <a href="https://arxiv.org/abs/1608.08710" target="_blank">Pruning Filters for Efficient ConvNets</a> 
					</li>
					<li>
						自动剪裁 <a href="https://arxiv.org/abs/1802.03494" target="_blank">AMC: AutoML for Model Compression and Acceleration on Mobile Devices</a> 
					</li>
				</ul>
				<ul>
					<li>
						FPGM剪裁 <a href="https://arxiv.org/abs/1811.00250" target="_blank">Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration</a> 
					</li>
					<li>
						基于batch nrom scale的剪裁方法&nbsp;<span style="background-color:#FFFDFA;"><a href="https://openreview.net/forum?id=HJ94fqApW" target="_blank">Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers</a></span><span style="font-family:&quot;color:#2C3A4A;font-size:2rem;background-color:#FFFDFA;"><a href="https://openreview.net/forum?id=HJ94fqApW" target="_blank">&nbsp;</a><a href="https://openreview.net/forum?id=HJ94fqApW" target="_blank"><span id="__kindeditor_bookmark_end_147__"></span></a></span> 
					</li>
				</ul>
			</td>
			<td>
				<br />
			</td>
		</tr>
		<tr>
			<td style="text-align:center;">
				量化
			</td>
			<td>
				<ul>
					<li>
						量化训练（QAT）<a href="https://arxiv.org/abs/1806.08342" target="_blank">Quantizing deep convolutional networks for efficient inference: A whitepaper</a> 
					</li>
					<li>
						离线量化（Post Training）<a href="http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf" target="_blank">原理</a> 
					</li>
					<li>
						Embedding量化&nbsp;
					</li>
					<li>
						DSQ: <a href="https://arxiv.org/abs/1908.05033" target="_blank">Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks</a> 
					</li>
					<li>
						PACT:&nbsp; <a href="https://arxiv.org/abs/1805.06085" target="_blank">PACT: Parameterized Clipping Activation for Quantized Neural Networks</a> 
					</li>
				</ul>
			</td>
			<td>
				<br />
			</td>
		</tr>
		<tr>
			<td style="text-align:center;">
				蒸馏
			</td>
			<td>
				<ul>
					<li>
						知识蒸馏 <a href="https://arxiv.org/abs/1503.02531" target="_blank">Distilling the Knowledge in a Neural Network</a> 
					</li>
					<li>
						FSP蒸馏&nbsp;<a href="http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf" target="_blank">A Gift from Knowledge Distillation:
Fast Optimization, Network Minimization and Transfer Learning</a> 
					</li>
				</ul>
				<ul>
					<li>
						YOLO蒸馏&nbsp;<a href="http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Mehta_Object_detection_at_200_Frames_Per_Second_ECCVW_2018_paper.pdf" target="_blank">Object detection at 200 Frames Per Second</a> 
					</li>
					<li>
						DML蒸馏 <a href="https://arxiv.org/abs/1706.00384" target="_blank">Deep Mutual Learning</a> 
					</li>
				</ul>
			</td>
			<td>
				<br />
			</td>
		</tr>
		<tr>
			<td style="text-align:center;">
				模型结构搜索(NAS)
			</td>
			<td>
				<ul>
					<li>
						Simulate Anneal NAS 原理
					</li>
					<li>
						DARTS <a href="https://arxiv.org/abs/1806.09055" target="_blank">DARTS: Differentiable Architecture Search</a> 
					</li>
				</ul>
				<ul>
					<li>
						PC-DARTS <a href="https://arxiv.org/abs/1907.05737" target="_blank">PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search</a> 
					</li>
					<li>
						OneShot&nbsp;
					</li>
				</ul>
				<p>
					<br />
				</p>
			</td>
			<td>
				<br />
			</td>
		</tr>
	</tbody>
</table>
<br />


## 安装

依赖：

Paddle >= 1.7.0

```bash
pip install paddleslim -i https://pypi.org/simple
```

## 使用

- [快速开始](docs/zh_cn/quick_start)：通过简单示例介绍如何快速使用PaddleSlim。
- [进阶教程](docs/zh_cn/tutorials)：PaddleSlim高阶教程。
- [模型库](docs/zh_cn/model_zoo.md)：各个压缩策略在图像分类、目标检测和图像语义分割模型上的实验结论，包括模型精度、预测速度和可供下载的预训练模型。
- [API文档](https://paddlepaddle.github.io/PaddleSlim/api_cn/index.html)
- [算法原理](https://paddlepaddle.github.io/PaddleSlim/algo/algo.html): 介绍量化、剪枝、蒸馏、NAS的基本知识背景。
- [Paddle检测库](https://github.com/PaddlePaddle/PaddleDetection/tree/master/slim)：介绍如何在检测库中使用PaddleSlim。
- [Paddle分割库](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/slim)：介绍如何在分割库中使用PaddleSlim。
- [PaddleLite](https://paddlepaddle.github.io/Paddle-Lite/)：介绍如何使用预测库PaddleLite部署PaddleSlim产出的模型。

## 部分压缩策略效果

### 分类模型

数据: ImageNet2012; 模型: MobileNetV1;

|压缩策略 |精度收益(baseline: 70.91%) |模型大小(baseline: 17.0M)|
|:---:|:---:|:---:|
| 知识蒸馏(ResNet50)| [+1.06%]() |-|
| 知识蒸馏(ResNet50) + int8量化训练 |[+1.10%]()| [-71.76%]()|
| 剪裁(FLOPs-50%) + int8量化训练|[-1.71%]()|[-86.47%]()|


### 图像检测模型

#### 数据：Pascal VOC；模型：MobileNet-V1-YOLOv3

|        压缩方法           | mAP(baseline: 76.2%)         | 模型大小(baseline: 94MB)      |
| :---------------------:   | :------------: | :------------:|
| 知识蒸馏(ResNet34-YOLOv3) | [+2.8%](#)      |       -       |
| 剪裁 FLOPs -52.88%        | [+1.4%]()      | [-67.76%]()   |
|知识蒸馏(ResNet34-YOLOv3)+剪裁(FLOPs-69.57%)| [+2.6%]()|[-67.00%]()|


#### 数据：COCO；模型：MobileNet-V1-YOLOv3

|        压缩方法           | mAP(baseline: 29.3%) | 模型大小|
| :---------------------:   | :------------: | :------:|
| 知识蒸馏(ResNet34-YOLOv3) |  [+2.1%]()     |-|
| 知识蒸馏(ResNet34-YOLOv3)+剪裁(FLOPs-67.56%) | [-0.3%]() | [-66.90%]()|

### 搜索

数据：ImageNet2012; 模型：MobileNetV2

|硬件环境           | 推理耗时 | Top1准确率(baseline:71.90%) |
|:---------------:|:---------:|:--------------------:|
| RK3288  | [-23%]()    | +0.07%    |
| Android cellphone  | [-20%]()    | +0.16% |
| iPhone 6s   | [-17%]()    | +0.32%  |
