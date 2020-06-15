# PaddleSlim

中文 | [English](README_en.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddleslim.readthedocs.io/en/latest/)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddleslim.readthedocs.io/zh_CN/latest/)
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
				<ul>
					<li>
						Sensitivity&nbsp;&nbsp;Pruner:&nbsp;<a href="https://arxiv.org/abs/1608.08710" target="_blank"><span style="font-family:&quot;font-size:14px;background-color:#FFFFFF;"><span style="font-family:&quot;font-size:14px;background-color:#FFFFFF;">Li H , Kadav A , Durdanovic I , et al. Pruning Filters for Efficient ConvNets[J]. 2016.</span></span></a>
					</li>
					<li>
						AMC Pruner:&nbsp;<a href="https://arxiv.org/abs/1802.03494" target="_blank"><span style="font-family:&quot;font-size:13px;background-color:#FFFFFF;">He, Yihui , et al. "AMC: AutoML for Model Compression and Acceleration on Mobile Devices." (2018).</span></a>
					</li>
					<li>
						FFPGM Pruner:&nbsp;<a href="https://arxiv.org/abs/1811.00250" target="_blank"><span style="font-family:&quot;font-size:14px;background-color:#FFFFFF;">He Y , Liu P , Wang Z , et al. Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration[C]// IEEE/CVF Conference on Computer Vision &amp; Pattern Recognition. IEEE, 2019.</span></a>
					</li>
					<li>
						Slim Pruner:<span style="background-color:#FFFDFA;">&nbsp;<a href="https://arxiv.org/pdf/1708.06519.pdf" target="_blank"><span style="font-family:&quot;font-size:14px;background-color:#FFFFFF;">Liu Z , Li J , Shen Z , et al. Learning Efficient Convolutional Networks through Network Slimming[J]. 2017.</span></a></span>
					</li>
					<li>
						<span style="background-color:#FFFDFA;">Opt Slim Pruner:&nbsp;<a href="https://arxiv.org/pdf/1708.06519.pdf" target="_blank"><span style="font-family:&quot;font-size:14px;background-color:#FFFFFF;">Ye Y , You G , Fwu J K , et al. Channel Pruning via Optimal Thresholding[J]. 2020.</span></a><br />
</span>
					</li>
				</ul>
			</td>
			<td>
					<ul>
						<li>
							<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/prune_api.rst" target="_blank">剪裁模块API文档</a>
						</li>
					        <li>
								<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/quick_start/pruning_tutorial.md" target="_blank">剪裁快速开始示例</a>
						</li>
						<li>
							<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/image_classification_sensitivity_analysis_tutorial.md" target="_blank">分类模敏感度分析教程</a>
						</li>
						<li>
							<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/paddledetection_slim_pruing_tutorial.md" target="_blank">检测模型剪裁教程</a>
						</li>
						<li>
								<span id="__kindeditor_bookmark_start_313__"></span><a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/paddledetection_slim_prune_dist_tutorial.md" target="_blank">检测模型剪裁+蒸馏教程</a>
						</li>
						<li>
								<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/paddledetection_slim_sensitivy_tutorial.md" target="_blank">检测模型敏感度分析教程</a>
						</li>
					</ul>
			</td>
		</tr>
		<tr>
			<td style="text-align:center;">
				量化
			</td>
			<td>
				<ul>
					<li>
						Quantization Aware Training:&nbsp;<a href="https://arxiv.org/abs/1806.08342" target="_blank"><span style="font-family:&quot;font-size:14px;background-color:#FFFFFF;">Krishnamoorthi R . Quantizing deep convolutional networks for efficient inference: A whitepaper[J]. 2018.</span></a>
					</li>
					<li>
						Post Training&nbsp;<span>Quantization&nbsp;</span><a href="http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf" target="_blank">原理</a>
					</li>
					<li>
						Embedding&nbsp;<span>Quantization:&nbsp;<a href="https://arxiv.org/pdf/1603.01025.pdf" target="_blank"><span style="font-family:&quot;font-size:14px;background-color:#FFFFFF;">Miyashita D , Lee E H , Murmann B . Convolutional Neural Networks using Logarithmic Data Representation[J]. 2016.</span></a></span>
					</li>
					<li>
						DSQ: <a href="https://arxiv.org/abs/1908.05033" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Gong, Ruihao, et al. "Differentiable soft quantization: Bridging full-precision and low-bit neural networks."&nbsp;</span><i>Proceedings of the IEEE International Conference on Computer Vision</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">. 2019.</span></a>
					</li>
					<li>
						PACT:&nbsp; <a href="https://arxiv.org/abs/1805.06085" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Choi, Jungwook, et al. "Pact: Parameterized clipping activation for quantized neural networks."&nbsp;</span><i>arXiv preprint arXiv:1805.06085</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">&nbsp;(2018).</span></a>
					</li>
				</ul>
			</td>
			<td>
				<ul>
					<li>
						<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/quantization_api.rst" target="_blank">量化API文档</a>
					</li>
					<li>
						<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/quick_start/quant_aware_tutorial.md" target="_blank">量化训练快速开始示例</a>
					</li>
					<li>
						<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/quick_start/quant_post_static_tutorial.md" target="_blank">静态离线量化快速开始示例</a>
					</li>
					<li>
						<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/paddledetection_slim_quantization_tutorial.md" target="_blank">检测模型量化教程</a>
					</li>
				</ul>
			</td>
		</tr>
		<tr>
			<td style="text-align:center;">
				蒸馏
			</td>
			<td>
				<ul>
					<li>
						<span>Knowledge Distillation</span>:&nbsp;<a href="https://arxiv.org/abs/1503.02531" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network."&nbsp;</span><i>arXiv preprint arXiv:1503.02531</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">&nbsp;(2015).</span></a>
					</li>
					<li>
						FSP <span>Knowledge Distillation</span>:&nbsp;&nbsp;<a href="http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Yim, Junho, et al. "A gift from knowledge distillation: Fast optimization, network minimization and transfer learning."&nbsp;</span><i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">. 2017.</span></a>
					</li>
					<li>
						YOLO Knowledge Distillation:&nbsp;&nbsp;<a href="http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Mehta_Object_detection_at_200_Frames_Per_Second_ECCVW_2018_paper.pdf" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Mehta, Rakesh, and Cemalettin Ozturk. "Object detection at 200 frames per second."&nbsp;</span><i>Proceedings of the European Conference on Computer Vision (ECCV)</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">. 2018.</span></a>
					</li>
					<li>
						DML:&nbsp;<a href="https://arxiv.org/abs/1706.00384" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Zhang, Ying, et al. "Deep mutual learning."&nbsp;</span><i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">. 2018.</span></a>
					</li>
				</ul>
			</td>
			<td>
				<ul>
					<li>
						<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/single_distiller_api.rst" target="_blank">蒸馏API文档</a>
					</li>
					<li>
						<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/quick_start/distillation_tutorial.md" target="_blank">蒸馏快速开始示例</a>
					</li>
					<li>
						<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/paddledetection_slim_distillation_tutorial.md" target="_blank">检测模型蒸馏教程</a>
					</li>
				</ul>
			</td>
		</tr>
		<tr>
			<td style="text-align:center;">
				模型结构搜索(NAS)
			</td>
			<td>
				<ul>
					<li>
						Simulate Anneal NAS:&nbsp;<a href="https://arxiv.org/pdf/2005.04117.pdf" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Abdelhamed, Abdelrahman, et al. "Ntire 2020 challenge on real image denoising: Dataset, methods and results."&nbsp;</span><i>The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">. Vol. 2. 2020.</span></a>
					</li>
					<li>
						DARTS <a href="https://arxiv.org/abs/1806.09055" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search."&nbsp;</span><i>arXiv preprint arXiv:1806.09055</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">&nbsp;(2018).</span></a>
					</li>
					<li>
						PC-DARTS <a href="https://arxiv.org/abs/1907.05737" target="_blank"><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">Xu, Yuhui, et al. "Pc-darts: Partial channel connections for memory-efficient differentiable architecture search."&nbsp;</span><i>arXiv preprint arXiv:1907.05737</i><span style="color:#222222;font-family:Arial, sans-serif;font-size:13px;background-color:#FFFFFF;">&nbsp;(2019).</span></a>
					</li>
					<li>
						OneShot&nbsp;
					</li>
				</ul>
			</td>
			<td>
						<ul>
							<li>
								<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/nas_api.rst" target="_blank">NAS API文档</a>
							</li>
							<li>
								<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/darts.rst" target="_blank">DARTS API文档</a>
							</li>
							<li>
								<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/quick_start/nas_tutorial.md" target="_blank">NAS快速开始示例</a>
							</li>
							<li>
								<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/paddledetection_slim_nas_tutorial.md" target="_blank">检测模型NAS教程</a>
							</li>
							<li>
								<a href="https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/sanas_darts_space.md" target="_blank">SANAS进阶版实验教程-压缩DARTS产出模型</a>
							</li>
						</ul>
			</td>
		</tr>
	</tbody>
</table>
<br />

## 安装

```bash
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 量化和Paddle版本的对应关系

如果在ARM和GPU上预测，每个版本都可以，如果在CPU上预测，请选择Paddle 2.0对应的PaddleSlim 1.1.0版本

- Paddle 1.7 系列版本，需要安装PaddleSlim 1.0.1版本

```bash
pip install paddleslim==1.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Paddle 1.8 系列版本，需要安装PaddleSlim 1.1.1版本

```bash
pip install paddleslim==1.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Paddle 2.0 系列版本，需要安装PaddleSlim 1.1.0版本

```bash
pip install paddleslim==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
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

## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

## 如何贡献代码

我们非常欢迎你可以为PaddleSlim提供代码，也十分感谢你的反馈。
