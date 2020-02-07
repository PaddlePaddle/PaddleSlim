# 模型库

## 1. 图象分类

数据集：ImageNet1000类

### 1.1 量化

| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型体积（MB） | TensorRT时延(V100, ms) | 下载 |
|:--:|:---:|:--:|:--:|:--:|:--:|
|MobileNetV1|-|70.99%/89.68%| 17 | -| [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) |
|MobileNetV1|quant_post|70.18%/89.25% (-0.81%/-0.43%)| 4.4 | - | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_quant_post.tar) |
|MobileNetV1|quant_aware|70.60%/89.57% (-0.39%/-0.11%)| 4.4 | -| [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_quant_aware.tar) |
| MobileNetV2 | - |72.15%/90.65%| 15 | - | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) |
| MobileNetV2 | quant_post | 71.15%/90.11% (-1%/-0.54%)| 4.0   | - | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV2_quant_post.tar) |
| MobileNetV2 | quant_aware |72.05%/90.63% (-0.1%/-0.02%)| 4.0 | - | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV2_quant_aware.tar) |
|ResNet50|-|76.50%/93.00%| 99 | 2.71 | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) |
|ResNet50|quant_post|76.33%/93.02% (-0.17%/+0.02%)| 25.1| 1.19 | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet50_quant_post.tar) |
|ResNet50|quant_aware|    76.48%/93.11% (-0.02%/+0.11%)| 25.1 | 1.17 | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet50_quant_awre.tar) |

<table border=0 cellpadding=0 cellspacing=0 width=861 style='border-collapse:
 collapse;table-layout:fixed;width:644pt'>
 <col width=87 style='width:65pt'>
 <col width=124 style='mso-width-source:userset;mso-width-alt:3968;width:93pt'>
 <col width=128 style='mso-width-source:userset;mso-width-alt:4096;width:96pt'>
 <col width=87 span=6 style='width:65pt'>
 <tr height=21 style='height:16.0pt'>
  <td colspan=3 height=21 class=xl63 width=339 style='height:16.0pt;width:254pt'>分类模型Lite时延(ms)</td>
  <td colspan=3 class=xl63 width=261 style='width:195pt'>armv7</td>
  <td colspan=3 class=xl63 width=261 style='width:195pt'>armv8</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>设备</td>
  <td class=xl63>模型类型</td>
  <td class=xl63>压缩策略</td>
  <td class=xl63>Thread 1</td>
  <td class=xl63>Thread 2</td>
  <td class=xl63>Thread 4</td>
  <td class=xl63>Thread 1</td>
  <td class=xl63>Thread 2</td>
  <td class=xl63>Thread 4</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=9 height=189 class=xl63 style='height:144.0pt'>高通835</td>
  <td rowspan=3 class=xl63>MobileNetV1</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>96.1942</td>
  <td class=xl63>53.2058</td>
  <td class=xl63>32.4468</td>
  <td class=xl63>88.4955</td>
  <td class=xl63>47.95</td>
  <td class=xl63>27.5189</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>60.8186</td>
  <td class=xl63>32.1931</td>
  <td class=xl63>16.4275</td>
  <td class=xl63>56.4311</td>
  <td class=xl63>29.5446</td>
  <td class=xl63>15.1053</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>60.5615</td>
  <td class=xl63>32.4016</td>
  <td class=xl63>16.6596</td>
  <td class=xl63>56.5266</td>
  <td class=xl63>29.7178</td>
  <td class=xl63>15.1459</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl63 style='height:48.0pt'>MobileNetV2</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>65.715</td>
  <td class=xl63>38.1346</td>
  <td class=xl63>25.155</td>
  <td class=xl63>61.3593</td>
  <td class=xl63>36.2038</td>
  <td class=xl63>22.849</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>48.3655</td>
  <td class=xl63>30.2021</td>
  <td class=xl63>21.9303</td>
  <td class=xl63>46.1487</td>
  <td class=xl63>27.3146</td>
  <td class=xl63>18.3053</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>48.3495</td>
  <td class=xl63>30.3069</td>
  <td class=xl63>22.1506</td>
  <td class=xl63>45.8715</td>
  <td class=xl63>27.4105</td>
  <td class=xl63>18.2223</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl63 style='height:48.0pt'>ResNet50</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>526.811</td>
  <td class=xl63>319.6486</td>
  <td class=xl63>205.8345</td>
  <td class=xl63>506.1138</td>
  <td class=xl63>335.1584</td>
  <td class=xl63>214.8936</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>475.4538</td>
  <td class=xl63>256.8672</td>
  <td class=xl63>139.699</td>
  <td class=xl63>461.7344</td>
  <td class=xl63>247.9506</td>
  <td class=xl63>145.9847</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>476.0507</td>
  <td class=xl63>256.5963</td>
  <td class=xl63>139.7266</td>
  <td class=xl63>461.9176</td>
  <td class=xl63>248.3795</td>
  <td class=xl63>149.353</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=9 height=189 class=xl63 style='height:144.0pt'>高通855</td>
  <td rowspan=3 class=xl63>MobileNetV1</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>33.5086</td>
  <td class=xl63>19.5773</td>
  <td class=xl63>11.7534</td>
  <td class=xl63>31.3474</td>
  <td class=xl63>18.5382</td>
  <td class=xl63>10.0811</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>36.7067</td>
  <td class=xl63>21.628</td>
  <td class=xl63>11.0372</td>
  <td class=xl63>14.0238</td>
  <td class=xl63>8.199</td>
  <td class=xl63>4.2588</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>37.0498</td>
  <td class=xl63>21.7081</td>
  <td class=xl63>11.0779</td>
  <td class=xl63>14.0947</td>
  <td class=xl63>8.1926</td>
  <td class=xl63>4.2934</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl63 style='height:48.0pt'>MobileNetV2</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>25.0396</td>
  <td class=xl63>15.2862</td>
  <td class=xl63>9.6609</td>
  <td class=xl63>22.909</td>
  <td class=xl63>14.1797</td>
  <td class=xl63>8.8325</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>28.1583</td>
  <td class=xl63>18.3317</td>
  <td class=xl63>11.8103</td>
  <td class=xl63>16.9158</td>
  <td class=xl63>11.1606</td>
  <td class=xl63>7.4148</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>28.1631</td>
  <td class=xl63>18.3917</td>
  <td class=xl63>11.8333</td>
  <td class=xl63>16.9399</td>
  <td class=xl63>11.1772</td>
  <td class=xl63>7.4176</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl63 style='height:48.0pt'>ResNet50</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>185.3705</td>
  <td class=xl63>113.0825</td>
  <td class=xl63>87.0741</td>
  <td class=xl63>177.7367</td>
  <td class=xl63>110.0433</td>
  <td class=xl63>74.4114</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>327.6883</td>
  <td class=xl63>202.4536</td>
  <td class=xl63>106.243</td>
  <td class=xl63>243.5621</td>
  <td class=xl63>150.0542</td>
  <td class=xl63>78.4205</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>328.2683</td>
  <td class=xl63>201.9937</td>
  <td class=xl63>106.744</td>
  <td class=xl63>242.6397</td>
  <td class=xl63>150.0338</td>
  <td class=xl63>79.8659</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=9 height=189 class=xl63 style='height:144.0pt'>麒麟970</td>
  <td rowspan=3 class=xl63>MobileNetV1</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>101.2455</td>
  <td class=xl63>56.4053</td>
  <td class=xl63>35.6484</td>
  <td class=xl63>94.8985</td>
  <td class=xl63>51.7251</td>
  <td class=xl63>31.9511</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>62.5012</td>
  <td class=xl63>32.1863</td>
  <td class=xl63>16.6018</td>
  <td class=xl63>57.7477</td>
  <td class=xl63>29.2116</td>
  <td class=xl63>15.0703</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>62.4412</td>
  <td class=xl63>32.2585</td>
  <td class=xl63>16.6215</td>
  <td class=xl63>57.825</td>
  <td class=xl63>29.2573</td>
  <td class=xl63>15.1206</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl63 style='height:48.0pt'>MobileNetV2</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>70.4176</td>
  <td class=xl63>42.0795</td>
  <td class=xl63>25.1939</td>
  <td class=xl63>68.9597</td>
  <td class=xl63>39.2145</td>
  <td class=xl63>22.6617</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>52.9961</td>
  <td class=xl63>31.5323</td>
  <td class=xl63>22.1447</td>
  <td class=xl63>49.4858</td>
  <td class=xl63>28.0856</td>
  <td class=xl63>18.7287</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>53.0961</td>
  <td class=xl63>31.7987</td>
  <td class=xl63>21.8334</td>
  <td class=xl63>49.383</td>
  <td class=xl63>28.2358</td>
  <td class=xl63>18.3642</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl63 style='height:48.0pt'>ResNet50</td>
  <td class=xl63>FP32 baseline</td>
  <td class=xl63>586.8943</td>
  <td class=xl63>344.0858</td>
  <td class=xl63>228.2293</td>
  <td class=xl63>573.3344</td>
  <td class=xl63>351.4332</td>
  <td class=xl63>225.8006</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_aware</td>
  <td class=xl63>488.361</td>
  <td class=xl63>260.1697</td>
  <td class=xl63>142.416</td>
  <td class=xl63>479.5668</td>
  <td class=xl63>249.8485</td>
  <td class=xl63>138.1742</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl63 style='height:16.0pt'>quant_post</td>
  <td class=xl63>489.6188</td>
  <td class=xl63>258.3279</td>
  <td class=xl63>142.6063</td>
  <td class=xl63>480.0064</td>
  <td class=xl63>249.5339</td>
  <td class=xl63>138.5284</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=87 style='width:65pt'></td>
  <td width=124 style='width:93pt'></td>
  <td width=128 style='width:96pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
 </tr>
 <![endif]>
</table>



### 1.2 剪裁


| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型体积（MB） | GFLOPs | 下载 |
|:--:|:---:|:--:|:--:|:--:|:--:|
| MobileNetV1 |    Baseline    |         70.99%/89.68%         |       17       |  1.11  | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) |
| MobileNetV1 |  uniform -50%  | 69.4%/88.66% (-1.59%/-1.02%)  |       9        |  0.56  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_uniform-50.tar) |
| MobileNetV1 | sensitive -30% |  70.4%/89.3% (-0.59%/-0.38%)  |       12       |  0.74  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_sensitive-30.tar) |
| MobileNetV1 | sensitive -50% | 69.8% / 88.9% (-1.19%/-0.78%) |       9        |  0.56  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_sensitive-50.tar) |
| MobileNetV2 |       -        |         72.15%/90.65%         |       15       |  0.59  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) |
| MobileNetV2 |  uniform -50%  | 65.79%/86.11% (-6.35%/-4.47%) |       11       | 0.296  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV2_uniform-50.tar) |
|  ResNet34   |       -        |         72.15%/90.65%         |       84       |  7.36  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) |
|  ResNet34   |  uniform -50%  | 70.99%/89.95% (-1.36%/-0.87%) |       41       |  3.67  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet34_uniform-50.tar) |
|  ResNet34   |  auto -55.05%  | 70.24%/89.63% (-2.04%/-1.06%) |       33       |  3.31  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet34_auto-55.tar) |




### 1.3 蒸馏

| 模型 | 压缩方法 | Top-1/Top-5 Acc | 模型体积（MB） | 下载 |
|:--:|:---:|:--:|:--:|:--:|
| MobileNetV1 |                     student                     |  70.99%/89.68%  |       17       | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) |
|ResNet50_vd|teacher|79.12%/94.44%| 99 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) |
|MobileNetV1|ResNet50_vd<sup>[1](#trans1)</sup> distill|72.77%/90.68% (+1.78%/+1.00%)| 17 | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV1_distilled.tar) |
| MobileNetV2 |                     student                     |  72.15%/90.65%  |       15       | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) |
| MobileNetV2 |            ResNet50_vd distill             |  74.28%/91.53% (+2.13%/+0.88%)  |       15       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/MobileNetV2_distilled.tar) |
|  ResNet50   |                     student                     |  76.50%/93.00%  |       99       | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) |
|ResNet101|teacher|77.56%/93.64%| 173 | [下载链接](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) |
|  ResNet50   |             ResNet101 distill              |  77.29%/93.65% (+0.79%/+0.65%)  |       99       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/ResNet50_distilled.tar) |

!!! note "Note"

    <a name="trans1">[1]</a>：带_vd后缀代表该预训练模型使用了Mixup，Mixup相关介绍参考[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)


## 2. 目标检测

### 2.1 量化

数据集： COCO 2017

|              模型              |  压缩方法   | 数据集 | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型体积（MB） |   TensorRT时延(V100, ms) |  下载     |
| :----------------------------: | :---------: | :----: | :-------: | :------------: | :------------: | :------------: | :------------: | :----------: |:----------: |
|      MobileNet-V1-YOLOv3       |      -      |  COCO  |     8     |      29.3      |      29.3      |      27.1      |       95       |  - | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
|      MobileNet-V1-YOLOv3       | quant_post  |  COCO  |     8     |     27.9 (-1.4)|    28.0 (-1.3)      |    26.0 (-1.0) |       25       | -  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_quant_post.tar) |
|      MobileNet-V1-YOLOv3       | quant_aware |  COCO  |     8     |     28.1 (-1.2)|  28.2 (-1.1)      |    25.8 (-1.2) |       26.3     | -  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_coco_quant_aware.tar) |
|      R34-YOLOv3                |      -      |  COCO  |     8     |      36.2      |      34.3      |      31.4      |       162       |  - | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
|      R34-YOLOv3                | quant_post  |  COCO  |     8     | 35.7 (-0.5)    |      -         |      -         |       42.7      |  - | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_post.tar) |
|      R34-YOLOv3                | quant_aware |  COCO  |     8     |  35.2 (-1.0)   | 33.3 (-1.0)    |     30.3 (-1.1)|       44       |  - | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r34_coco_quant_aware.tar) |
| R50-dcn-YOLOv3 obj365_pretrain |      -      |  COCO  |     8     |      41.4      |       -      |       -       |       177       | 18.56  |[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_obj365_pretrained_coco.tar) |
| R50-dcn-YOLOv3 obj365_pretrain | quant_aware |  COCO  |     8     |   40.6 (-0.8)  |       37.5   |       34.1    |       66       |  14.64 | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_quant_aware.tar) |



数据集：WIDER-FACE



|      模型      |  压缩方法   | Image/GPU | 输入尺寸 |        Easy/Medium/Hard         | 模型体积（MB） |                             下载                             |
| :------------: | :---------: | :-------: | :------: | :-----------------------------: | :------------: | :----------------------------------------------------------: |
|   BlazeFace    |      -      |     8     |   640    |         91.5/89.2/79.7          |      815       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_original.tar) |
|   BlazeFace    | quant_post  |     8     |   640    | 87.8/85.1/74.9 (-3.7/-4.1/-4.8) |      228       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_origin_quant_post.tar) |
|   BlazeFace    | quant_aware |     8     |   640    | 90.5/87.9/77.6 (-1.0/-1.3/-2.1) |      228       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_origin_quant_aware.tar) |
| BlazeFace-Lite |      -      |     8     |   640    |         90.9/88.5/78.1          |      711       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_lite.tar) |
| BlazeFace-Lite | quant_post  |     8     |   640    | 89.4/86.7/75.7 (-1.5/-1.8/-2.4) |      211       | [下载链接]((https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_lite_quant_post.tar)) |
| BlazeFace-Lite | quant_aware |     8     |   640    | 89.7/87.3/77.0 (-1.2/-1.2/-1.1) |      211       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_lite_quant_aware.tar) |
| BlazeFace-NAS  |      -      |     8     |   640    |         83.7/80.7/65.8          |      244       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_nas.tar) |
| BlazeFace-NAS  | quant_post  |     8     |   640    | 81.6/78.3/63.6 (-2.1/-2.4/-2.2) |       71       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_nas_quant_post.tar) |
| BlazeFace-NAS  | quant_aware |     8     |   640    | 83.1/79.7/64.2 (-0.6/-1.0/-1.6) |       71       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/blazeface_nas_quant_aware.tar) |

### 2.2 剪裁

数据集：Pasacl VOC & COCO 2017

|              模型              |     压缩方法      |   数据集   | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型体积(MB) | GFLOPs (608*608) |                             下载                             |
| :----------------------------: | :---------------: | :--------: | :-------: | :------------: | :------------: | :------------: | :----------: | :--------------: | :----------------------------------------------------------: |
|      MobileNet-V1-YOLOv3       |     Baseline      | Pascal VOC |     8     |      76.2      |      76.7      |      75.3      |      94      |      40.49       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
|      MobileNet-V1-YOLOv3       | sensitive -52.88% | Pascal VOC |     8     |  77.6 (+1.4)   |   77.7 (1.0)   |  75.5 (+0.2)   |      31      |      19.08       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_v1_voc_prune.tar) |
|      MobileNet-V1-YOLOv3       |         -         |    COCO    |     8     |      29.3      |      29.3      |      27.0      |      95      |      41.35       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
|      MobileNet-V1-YOLOv3       | sensitive -51.77% |    COCO    |     8     |  26.0 (-3.3)   |  25.1 (-4.2)   |  22.6 (-4.4)   |      32      |      19.94       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenet_v1_prune.tar) |
|         R50-dcn-YOLOv3         |         -         |    COCO    |     8     |      39.1      |       -        |       -        |     177      |      89.60       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar) |
|         R50-dcn-YOLOv3         | sensitive -9.37%  |    COCO    |     8     |  39.3 (+0.2)   |       -        |       -        |     150      |      81.20       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_prune.tar) |
|         R50-dcn-YOLOv3         | sensitive -24.68% |    COCO    |     8     |  37.3 (-1.8)   |       -        |       -        |     113      |      67.48       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_prune578.tar) |
| R50-dcn-YOLOv3 obj365_pretrain |         -         |    COCO    |     8     |      41.4      |       -        |       -        |     177      |      89.60       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn_obj365_pretrained_coco.tar) |
| R50-dcn-YOLOv3 obj365_pretrain | sensitive -9.37%  |    COCO    |     8     |  40.5 (-0.9)   |       -        |       -        |     150      |      81.20       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_prune.tar) |
| R50-dcn-YOLOv3 obj365_pretrain | sensitive -24.68% |    COCO    |     8     |  37.8 (-3.3)   |       -        |       -        |     113      |      67.48       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_r50vd_dcn_obj365_pretrained_coco_prune578.tar) |

### 2.3 蒸馏

数据集：Pasacl VOC & COCO 2017


|        模型         |        压缩方法         |   数据集   | Image/GPU | 输入608 Box AP | 输入416 Box AP | 输入320 Box AP | 模型体积（MB） |                             下载                             |
| :-----------------: | :---------------------: | :--------: | :-------: | :------------: | :------------: | :------------: | :------------: | :----------------------------------------------------------: |
| MobileNet-V1-YOLOv3 |            -            | Pascal VOC |     8     |      76.2      |      76.7      |      75.3      |       94       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar) |
|   ResNet34-YOLOv3   |            -            | Pascal VOC |     8     |      82.6      |      81.9      |      80.1      |      162       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar) |
| MobileNet-V1-YOLOv3 | ResNet34-YOLOv3 distill | Pascal VOC |     8     |  79.0 (+2.8)   |  78.2 (+1.5)   |  75.5 (+0.2)   |       94       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_voc_distilled.tar) |
| MobileNet-V1-YOLOv3 |            -            |    COCO    |     8     |      29.3      |      29.3      |      27.0      |       95       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |
|   ResNet34-YOLOv3   |            -            |    COCO    |     8     |      36.2      |      34.3      |      31.4      |      163       | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) |
| MobileNet-V1-YOLOv3 | ResNet34-YOLOv3 distill |    COCO    |     8     |  31.4 (+2.1)   |  30.0 (+0.7)   |  27.1 (+0.1)   |       95       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/yolov3_mobilenetv1_coco_distilled.tar) |


## 3. 图像分割

数据集：Cityscapes

### 3.1 量化

|          模型          |  压缩方法   |     mIoU      | 模型体积（MB） |                             下载                             |
| :--------------------: | :---------: | :-----------: | :------------: | :----------------------------------------------------------: |
| DeepLabv3+/MobileNetv1 |      -      |     63.26     |      6.6       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/deeplabv3_mobilenetv1.tar )                         |
| DeepLabv3+/MobileNetv1 | quant_post  | 58.63 (-4.63) |      1.8       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/deeplabv3_mobilenetv1_2049x1025_quant_post.tar) |
| DeepLabv3+/MobileNetv1 | quant_aware | 62.03 (-1.23) |      1.8       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/deeplabv3_mobilenetv1_2049x1025_quant_aware.tar) |
| DeepLabv3+/MobileNetv2 |      -      |     69.81     |      7.4       | [下载链接](https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz) |
| DeepLabv3+/MobileNetv2 | quant_post  | 67.59 (-2.22) |      2.1       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/deeplabv3_mobilenetv2_2049x1025_quant_post.tar) |
| DeepLabv3+/MobileNetv2 | quant_aware | 68.33 (-1.48) |      2.1       | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/deeplabv3_mobilenetv2_2049x1025_quant_aware.tar) |

<br/>

<table border=0 cellpadding=0 cellspacing=0 width=841 style='border-collapse:
 collapse;table-layout:fixed;width:629pt'>
 <col width=87 style='width:65pt'>
 <col width=105 style='mso-width-source:userset;mso-width-alt:3370;width:79pt'>
 <col width=127 style='mso-width-source:userset;mso-width-alt:4053;width:95pt'>
 <col width=87 span=6 style='width:65pt'>
 <tr height=21 style='height:16.0pt'>
  <td colspan=3 height=21 class=xl65 width=319 style='height:16.0pt;width:239pt'>图像分割模型Lite时延(ms),
  输入尺寸769x769</td>
  <td colspan=3 class=xl65 width=261 style='width:195pt'>armv7</td>
  <td colspan=3 class=xl65 width=261 style='width:195pt'>armv8</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>设备</td>
  <td class=xl65>模型类型</td>
  <td class=xl65>压缩策略</td>
  <td class=xl65>Thread 1</td>
  <td class=xl65>Thread 2</td>
  <td class=xl65>Thread 4</td>
  <td class=xl65>Thread 1</td>
  <td class=xl65>Thread 2</td>
  <td class=xl65>Thread 4</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=6 height=126 class=xl65 style='height:96.0pt'>高通835</td>
  <td rowspan=3 class=xl66 width=105 style='width:79pt'>Deeplabv3- MobileNetV1</td>
  <td class=xl65>FP32 baseline</td>
  <td class=xl65>1227.9894</td>
  <td class=xl65>734.1922</td>
  <td class=xl65>527.9592</td>
  <td class=xl65>1109.96</td>
  <td class=xl65>699.3818</td>
  <td class=xl65>479.0818</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_aware</td>
  <td class=xl65>848.6544</td>
  <td class=xl65>512.785</td>
  <td class=xl65>382.9915</td>
  <td class=xl65>752.3573</td>
  <td class=xl65>455.0901</td>
  <td class=xl65>307.8808</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_post</td>
  <td class=xl65>840.2323</td>
  <td class=xl65>510.103</td>
  <td class=xl65>371.9315</td>
  <td class=xl65>748.9401</td>
  <td class=xl65>452.1745</td>
  <td class=xl65>309.2084</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl66 width=105 style='height:48.0pt;width:79pt'>Deeplabv3-MobileNetV2</td>
  <td class=xl65>FP32 baseline</td>
  <td class=xl65>1282.8126</td>
  <td class=xl65>793.2064</td>
  <td class=xl65>653.6538</td>
  <td class=xl65>1193.9908</td>
  <td class=xl65>737.1827</td>
  <td class=xl65>593.4522</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_aware</td>
  <td class=xl65>976.0495</td>
  <td class=xl65>659.0541</td>
  <td class=xl65>513.4279</td>
  <td class=xl65>892.1468</td>
  <td class=xl65>582.9847</td>
  <td class=xl65>484.7512</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_post</td>
  <td class=xl65>981.44</td>
  <td class=xl65>658.4969</td>
  <td class=xl65>538.6166</td>
  <td class=xl65>885.3273</td>
  <td class=xl65>586.1284</td>
  <td class=xl65>484.0018</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=6 height=126 class=xl65 style='height:96.0pt'>高通855</td>
  <td rowspan=3 class=xl66 width=105 style='width:79pt'>Deeplabv3-MobileNetV1</td>
  <td class=xl65>FP32 baseline</td>
  <td class=xl65>568.8748</td>
  <td class=xl65>339.8578</td>
  <td class=xl65>278.6316</td>
  <td class=xl65>420.6031</td>
  <td class=xl65>281.3197</td>
  <td class=xl65>217.5222</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_aware</td>
  <td class=xl65>608.7578</td>
  <td class=xl65>347.2087</td>
  <td class=xl65>260.653</td>
  <td class=xl65>241.2394</td>
  <td class=xl65>177.3456</td>
  <td class=xl65>143.9178</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_post</td>
  <td class=xl65>609.0142</td>
  <td class=xl65>347.3784</td>
  <td class=xl65>259.9825</td>
  <td class=xl65>239.4103</td>
  <td class=xl65>180.1894</td>
  <td class=xl65>139.9178</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl66 width=105 style='height:48.0pt;width:79pt'>Deeplabv3-MobileNetV2</td>
  <td class=xl65>FP32 baseline</td>
  <td class=xl65>639.4425</td>
  <td class=xl65>390.1851</td>
  <td class=xl65>322.7014</td>
  <td class=xl65>477.7667</td>
  <td class=xl65>339.7411</td>
  <td class=xl65>262.2847</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_aware</td>
  <td class=xl65>703.7275</td>
  <td class=xl65>497.689</td>
  <td class=xl65>417.1296</td>
  <td class=xl65>394.3586</td>
  <td class=xl65>300.2503</td>
  <td class=xl65>239.9204</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_post</td>
  <td class=xl65>705.7589</td>
  <td class=xl65>474.4076</td>
  <td class=xl65>427.2951</td>
  <td class=xl65>394.8352</td>
  <td class=xl65>297.4035</td>
  <td class=xl65>264.6724</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=6 height=126 class=xl65 style='height:96.0pt'>麒麟970</td>
  <td rowspan=3 class=xl66 width=105 style='width:79pt'>Deeplabv3-MobileNetV1</td>
  <td class=xl65>FP32 baseline</td>
  <td class=xl65>1682.1792</td>
  <td class=xl65>1437.9774</td>
  <td class=xl65>1181.0246</td>
  <td class=xl65>1261.6739</td>
  <td class=xl65>1068.6537</td>
  <td class=xl65>690.8225</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_aware</td>
  <td class=xl65>1062.3394</td>
  <td class=xl65>1248.1014</td>
  <td class=xl65>878.3157</td>
  <td class=xl65>774.6356</td>
  <td class=xl65>710.6277</td>
  <td class=xl65>528.5376</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_post</td>
  <td class=xl65>1109.1917</td>
  <td class=xl65>1339.6218</td>
  <td class=xl65>866.3587</td>
  <td class=xl65>771.5164</td>
  <td class=xl65>716.5255</td>
  <td class=xl65>500.6497</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=3 height=63 class=xl66 width=105 style='height:48.0pt;width:79pt'>Deeplabv3-MobileNetV2</td>
  <td class=xl65>FP32 baseline</td>
  <td class=xl65>1771.1301</td>
  <td class=xl65>1746.0569</td>
  <td class=xl65>1222.4805</td>
  <td class=xl65>1448.9739</td>
  <td class=xl65>1192.4491</td>
  <td class=xl65>760.606</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_aware</td>
  <td class=xl65>1320.2905</td>
  <td class=xl65>921.4522</td>
  <td class=xl65>676.0732</td>
  <td class=xl65>1145.8801</td>
  <td class=xl65>821.5685</td>
  <td class=xl65>590.1713</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>quant_post</td>
  <td class=xl65>1320.386</td>
  <td class=xl65>918.5328</td>
  <td class=xl65>672.2481</td>
  <td class=xl65>1020.753</td>
  <td class=xl65>820.094</td>
  <td class=xl65>591.4114</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=87 style='width:65pt'></td>
  <td width=105 style='width:79pt'></td>
  <td width=127 style='width:95pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
 </tr>
 <![endif]>
</table>

### 3.2 剪裁

|   模型    |     压缩方法      |     mIoU      | 模型体积（MB） | GFLOPs |                             下载                             |
| :-------: | :---------------: | :-----------: | :------------: | :----: | :----------------------------------------------------------: |
| fast-scnn |     baseline      |     69.64     |       11       | 14.41  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/fast_scnn_cityscape.tar) |
| fast-scnn | uniform  -17.07%  | 69.58 (-0.06) |      8.5       | 11.95  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/fast_scnn_cityscape_uniform-17.tar) |
| fast-scnn | sensitive -47.60% | 66.68 (-2.96) |      5.7       |  7.55  | [下载链接](https://paddlemodels.bj.bcebos.com/PaddleSlim/fast_scnn_cityscape_sensitive-47.tar) |
