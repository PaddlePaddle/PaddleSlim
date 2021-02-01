# 人脸检测模型小模型结构搜索教程

教程内容请参考：https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/slim/nas/README.md

## 概述

我们选取人脸检测的BlazeFace模型作为神经网络搜索示例，该示例使用PaddleSlim 辅助完成神经网络搜索实验。
基于PaddleSlim进行搜索实验过程中，搜索限制条件可以选择是浮点运算数(FLOPs)限制还是硬件延时(latency)限制，硬件延时限制需要提供延时表。本示例提供一份基于blazeface搜索空间的硬件延时表，名称是latency_855.txt(基于PaddleLite在骁龙855上测试的延时)，可以直接用该表进行blazeface的硬件延时搜索实验。
