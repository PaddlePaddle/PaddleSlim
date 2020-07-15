# CPU上部署量化模型教程

在Intel(R) Xeon(R) Gold 6271机器上，经过量化和DNNL加速，INT8模型在单线程上性能为原FP32模型的3~4倍；在 Intel(R) Xeon(R) Gold 6148，单线程性能为原FP32模型的1.5倍，而精度仅有极小下降。图像分类量化的样例教程请参考[图像分类INT8模型在CPU优化部署和预测](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/mkldnn_quant/quant_aware/PaddleCV_mkldnn_quantaware_tutorial_cn.md)。自然语言处理模型的量化请参考[ERNIE INT8 模型精度与性能复现](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn)


## 图像分类INT8模型在 Xeon(R) 6271 上的精度和性能

>**图像分类INT8模型在 Intel(R) Xeon(R) Gold 6271 上精度**

|    Model     | FP32 Top1 Accuracy | INT8 Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

>**图像分类INT8模型在 Intel(R) Xeon(R) Gold 6271 单核上性能**

|    Model     | FP32 (images/s) | INT8 (images/s) | Ratio (INT8/FP32)  |
| :----------: | :-------------: | :-----------------: | :---------------:  |
| MobileNet-V1 |      74.05      |       196.98        |      2.66          |
| MobileNet-V2 |      88.60      |       187.67        |      2.12          |
|  ResNet101   |      7.20       |       26.43         |      3.67          |
|   ResNet50   |      13.23      |       47.44         |      3.59          |
|    VGG16     |      3.47       |       10.20         |      2.94          |
|    VGG19     |      2.83       |       8.67          |      3.06          |

## 自然语言处理INT8模型在 Xeon(R) 6271 上的精度和性能

>**I. Ernie INT8 DNNL 在 Intel(R) Xeon(R) Gold 6271 的精度结果**

|     Model    |  FP32 Accuracy | INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |          80.20%        |         79.44%   |     -0.76%      |  


>**II. Ernie INT8 DNNL 在 Intel(R) Xeon(R) Gold 6271 上单样本耗时**

|     Threads  | FP32 Latency (ms) | INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:-----------------:|
| 1 thread     |       237.21          |      79.26           |    2.99X       |
| 20 threads   |       22.08           |      12.57           |    1.76X       |
