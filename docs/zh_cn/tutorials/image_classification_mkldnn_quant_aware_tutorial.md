# 使用CPU进行图像分类模型定点量化教程

量化是模型压缩的重要手段。在PaddlePaddle中，量化策略`post`为使用离线量化得到模型，`aware`为在线量化训练得到模型。本教程介绍了在CPU上使用训练时量化策略(`aware`)，结合CPU MKL-DNN库，对图像分类模型进行量化和加速。在Intel(R) Xeon(R) Gold 6271机器上，经过量化和MKL-DNN加速，INT8模型在单线程上性能为原FP32模型的3~4倍，而精度仅有极小下降；在 Intel(R) Xeon(R) Gold 6148，单线程性能是原FP32模型的1.5倍。样例教程请参考：https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/mkldnn_quant/quant_aware/PaddleCV_mkldnn_quantaware_tutorial_cn.md。


# QAT量化图像分类模型在 Xeon(R) 6271 和 Xeon(R) 6148 上的精度和性能

>**I. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271的精度**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

>**II. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6148 的精度**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.85%         |   0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.08%         |   0.18%   |       90.56%       |         90.66%         |  +0.10%   |
|  ResNet101   |       77.50%       |         77.51%         |   0.01%   |       93.58%       |         93.50%         |  -0.08%   |
|   ResNet50   |       76.63%       |         76.55%         |  -0.08%   |       93.10%       |         92.96%         |  -0.14%   |
|    VGG16     |       72.08%       |         71.72%         |  -0.36%   |       90.63%       |         89.75%         |  -0.88%   |
|    VGG19     |       72.57%       |         72.08%         |  -0.49%   |       90.84%       |         90.11%         |  -0.73%   |

## QAT量化模型性能

>**I. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271的单核的性能**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      73.98      |       227.73        |       3.08        |
| MobileNet-V2 |      86.59      |       206.74        |       2.39        |
|  ResNet101   |      7.15       |        26.69        |       3.73        |
|   ResNet50   |      13.15      |        49.33        |       3.75        |
|    VGG16     |      3.34       |        10.15        |       3.04        |
|    VGG19     |      2.83       |        8.67         |       3.07        |


>**II. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6148的单核的性能**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      75.23      |       111.15        |       1.48        |
| MobileNet-V2 |      86.65      |       127.21        |       1.47        |
|  ResNet101   |      6.61       |        10.60        |       1.60        |
|   ResNet50   |      12.42      |        19.74        |       1.59        |
|    VGG16     |      3.31       |        4.74         |       1.43        |
|    VGG19     |      2.68       |        3.91         |       1.46        |
