低比特量化
================
定点量化是使用更少的比特数（如8-bit、3-bit、2-bit等）表示神经网络的权重和激活。

在线量化训练(QAT)
-------------
在线量化是在模型训练的过程中建模定点量化对模型的影响，通过在模型计算图中插入量化节点，在训练建模量化对模型精度的影响降低量化损失。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/quanter/qat.rst#qat)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_aware)

PACT
------------
PACT在量化激活值之前去掉一些离群点来使量化精度提高。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/quanter/qat.rst#qat)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.0.0/demo/quant/pact_quant_aware#%E5%AE%9A%E4%B9%89pact%E5%87%BD%E6%95%B0)

静态离线量化(PTQ Static)
------------
静态离线量化，使用少量校准数据计算量化因子，可以快速得到量化模型。使用该量化模型进行预测，可以减少计算量、降低计算内存、减小模型大小。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_post_static)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_post_static)

动态离线量化(PTQ Dynamic)
------------
动态离线量化，将模型中特定OP的权重从FP32类型量化成INT8/16类型。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_post_dynamic)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_post_dynamic)

Embedding量化
------------
针对 Embedding 参数进行量化。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_embedding)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/quant/quantization_api.rst#quant_embedding)

剪枝
================
剪裁通过从冗余模型中删除不重要的子网络来压缩模型。

敏感度剪枝
-----------
敏感度裁剪指的是通过各个层的敏感度分析来确定各个卷积层的剪裁率，需要和其他裁剪方法配合使用。以下链接仅指向L1Norm剪枝方法和敏感度剪枝混合使用，其他剪枝方法和敏感度剪枝混合使用的方式可以去相应方法的API链接里查看。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/tutorials/pruning/dygraph/filter_pruning.md#41-%E5%8D%B7%E7%A7%AF%E9%87%8D%E8%A6%81%E6%80%A7%E5%88%86%E6%9E%90)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/prune/prune_api.rst#sensitivity)

FPGM
------------
该策略通过统计Filters两两之间的几何距离来评估单个卷积内的Filters的重要性。直觉上理解，离其它Filters平均距离越远的Filter越重要。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/pruners/fpgm_filter_pruner.rst#fpgmfilterpruner)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/9b01b195f0c4bc34a1ab434751cb260e13d64d9e/docs/zh_cn/tutorials/pruning/overview.md#%E9%9D%99%E6%80%81%E5%9B%BE-1)

L1Norm
------------
该策略使用l1-norm统计量来表示一个卷积层内各个Filters的重要性，l1-norm越大的Filter越重要。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/pruners/l1norm_filter_pruner.rst#l1normfilterpruner)  
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/9b01b195f0c4bc34a1ab434751cb260e13d64d9e/docs/zh_cn/tutorials/pruning/overview.md#%E9%9D%99%E6%80%81%E5%9B%BE)

L2Norm
------------
该策略使用l2-norm统计量来表示一个卷积层内各个Filters的重要性，l2-norm越大的Filter越重要。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/pruners/l2norm_filter_pruner.rst#l2normfilterpruner)

SlimFilter
------------
该策略根据卷积之后的batch_norm的scale来评估当前卷积内各个Filters的重要性。scale越大，对应的Filter越重要。

[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/9b01b195f0c4bc34a1ab434751cb260e13d64d9e/docs/zh_cn/tutorials/pruning/overview.md#slimfilterpruner)

OptSlimFilter
------------
根据卷积层后链接的batch_norm层的scale参数计算出要裁剪的最优裁剪阈值，并根据该阈值进行通道裁剪。

[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/9b01b195f0c4bc34a1ab434751cb260e13d64d9e/docs/zh_cn/tutorials/pruning/overview.md#optslimfilterpruner)

模型结构搜索
================
模型结构搜索指的是定义一个搜索空间，其中包括所有候选神经网络结构，不断从中搜索最优网络结构的优化策略。

Once-For-All
------------
OFA是一种基于One-Shot NAS的压缩方案。这种方式比较高效，其优势是只需要训练一个超网络就可以从中选择满足不同延时要求的子模型。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/ofa/ofa_api.rst#ofa-%E8%AE%AD%E7%BB%83)

SANAS
------------
SANAS是基于模拟退火的方式进行网络结构搜索，在机器资源不多的情况下，选择这种方式一般能得到比强化学习更好的模型。

[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/nas/nas_api.rst#sanas)

RLNAS
------------
RLNAS是基于强化学习的方式进行网络结构搜索，这种方式需要耗费大量机器资源。

[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/nas/nas_api.rst#rlnas)

DARTS
------------
DARTS是基于梯度的方式进行网络结构搜索，可以大大缩短搜索时长。

[动态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/dygraph/nas/darts.rst#%E5%8F%AF%E5%BE%AE%E5%88%86%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E6%90%9C%E7%B4%A2darts)

PC-DARTS
------------
[动态图](https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.0.0/demo/darts#%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2)

Hardware-aware Search
------------
基于硬件进行模型结构搜索，减少搜索和实际部署上的差异。

[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/83fd35cd41bba6dbd3b2eb425b867d3d28836cb0/docs/zh_cn/api_cn/static/common/analysis_api.rst#tablelatencyevaluator)


蒸馏
================
模型蒸馏是将复杂网络中的有用信息将复杂网络中的有用信息提取出来提取出来，迁移到一个更小的网络中去。

FSP
------------
出自论文[A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf)

[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/dist/single_distiller_api.rst#fsp_loss)

DML
------------
出自论文[Deep Mutual Learning](https://arxiv.org/abs/1706.00384)

[静态图](https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.0.0/demo/deep_mutual_learning)

DK
------------
[静态图](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0.0/docs/zh_cn/api_cn/static/dist/single_distiller_api.rst#l2_loss)

