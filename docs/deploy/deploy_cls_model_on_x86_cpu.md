# PaddleSlim量化模型在CPU上的预测部署

本教程以图像分类模型为例，介绍在CPU上转化PaddleSlim产出的量化模型并部署和预测的流程。对于常见图像分类模型，在Casecade Lake机器上（例如Intel® Xeon® Gold 6271、6248，X2XX等），INT8模型进行推理的速度通常是FP32模型的3-3.7倍；在SkyLake机器（例如Intel® Xeon® Gold 6148、8180，X1XX等）上，使用INT8模型进行推理的速度通常是FP32模型的1.5倍。

本教程所需的代码较多，在此无法全面展示，具体例程请参考[mkldnn_quant](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/mkldnn_quant)
