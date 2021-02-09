# Intel CPU量化训练

在Intel Casecade Lake机器上（如：Intel(R) Xeon(R) Gold 6271），经过量化和DNNL加速，INT8模型在单线程上性能为FP32模型的3~3.7倍；在Intel SkyLake机器上（如：Intel(R) Xeon(R) Gold 6148），单线程性能为FP32模型的1.5倍，而精度仅有极小下降。图像分类量化的样例教程请参考[图像分类INT8模型在CPU优化部署和预测](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/mkldnn_quant/)。自然语言处理模型的量化请参考[ERNIE INT8 模型精度与性能复现](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn)

## 图像分类INT8模型在 Xeon(R) 6271 上的精度和性能

>**图像分类INT8模型在 Intel(R) Xeon(R) Gold 6271 上精度**

|     Model    | FP32 Top1 Accuracy | INT8 Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 Top5 Accuracy | Top5 Diff |
|:------------:|:------------------:|:------------------:|:---------:|:------------------:|:------------------:|:---------:|
| MobileNet-V1 |       70.78%       |       70.74%       |   -0.04%  |       89.69%       |       89.43%       |   -0.26%  |
| MobileNet-V2 |       71.90%       |       72.21%       |   0.31%   |       90.56%       |       90.62%       |   0.06%   |
|   ResNet101  |       77.50%       |       77.60%       |   0.10%   |       93.58%       |       93.55%       |   -0.03%  |
|   ResNet50   |       76.63%       |       76.50%       |   -0.13%  |       93.10%       |       92.98%       |   -0.12%  |
|     VGG16    |       72.08%       |       71.74%       |   -0.34%  |       90.63%       |       89.71%       |   -0.92%  |
|     VGG19    |       72.57%       |       72.12%       |   -0.45%  |       90.84%       |       90.15%       |   -0.69%  |

>**图像分类INT8模型在 Intel(R) Xeon(R) Gold 6271 单核上性能**

|     Model    | FP32 (images/s) | INT8 (images/s) | Ratio (INT8/FP32) |
|:------------:|:---------------:|:---------------:|:-----------------:|
| MobileNet-V1 |      74.05      |      216.36     |        2.92       |
| MobileNet-V2 |      88.60      |      205.84     |        2.32       |
|   ResNet101  |       7.20      |      26.48      |        3.68       |
|   ResNet50   |      13.23      |      50.02      |        3.78       |
|     VGG16    |       3.47      |      10.67      |        3.07       |
|     VGG19    |       2.83      |       9.09      |        3.21       |

## 自然语言处理INT8模型在 Xeon(R) 6271 上的精度和性能

>**I. Ernie INT8 DNNL 在 Intel(R) Xeon(R) Gold 6271 的精度结果**

| Model | FP32 Accuracy | INT8 Accuracy | Accuracy Diff |
| :---: | :-----------: | :-----------: | :-----------: |
| Ernie |    80.20%     |    79.44%     |    -0.76%     |


>**II. Ernie INT8 DNNL 在 Intel(R) Xeon(R) Gold 6271 上单样本耗时**

|  Threads   | FP32 Latency (ms) | INT8 Latency (ms) | Ratio (FP32/INT8) |
| :--------: | :---------------: | :---------------: | :---------------: |
|  1 thread  |      237.21       |       79.26       |       2.99X       |
| 20 threads |       22.08       |       12.57       |       1.76X       |
