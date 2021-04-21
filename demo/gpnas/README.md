# GP-NAS使用示例

CVPR2021_NAS_competition_gpnas_demo.py演示如何使用GP-NAS参加[CVPR 2021 NAS国际比赛](https://www.cvpr21-nas.com/competition) Track2 demo

[CVPR 2021 NAS国际比赛Track2 studio地址](https://aistudio.baidu.com/aistudio/competition/detail/71?lang=en)

[AI studio GP-NAS demo](https://aistudio.baidu.com/aistudio/projectdetail/1824958)

基于本demo的改进版方案可以获得双倍奖金

# CVPR 2021 NAS国际比赛背景

在不训练的情况下，准确的预测任意模型结构性能非常重要。基于此，我们不仅可以深度的分析怎样的模型结构会有很好的性能，怎样的模型性能会很差。同时还能够预测出满足任意硬件延时约束下的最优的模型结构。本赛事提供了部分（小样本）模型结构与模型精度之间对应关系的bench mark,参赛选手既可以通过黑盒的方式直接进行训练，也可以使用白盒的方式进行参数估计。

本赛道采用Mobilenet-like搜索空间，其中16个block可以搜索，每层的搜索空间由一个4元组[layer_index1, layer_index2, OP1, OP2]构成, layer1_index取值范围在[2,17]或为layer2_index除了[2,17]之外还可以为0，为0则表示本层只有一个后序节点，每层可以选择与1到2个编号大于该层数的后序节点相连接，OP1取值范围在[1,6]表示6种（kernel size三种选择，膨胀系数2种选择）不同的链接方式，OP2取值范围除了[1,6]之外还可以为0，为0则表示本层只有一个后序节点。

本赛道分为两个阶段，第一阶段为线性拓扑，即每层只与编号比本层多1层的后续节点相连接，第二阶段在第一阶段基础至上考察模型的few shot能力。

第二阶段赛题背景: 为什么关注acc而非ranking? 在很多场景，我们需要搜索到在特定硬件上精度不低于特定指标的最优的模型结构，只预测相对排序无法保证搜索结构可以满足精度的约束条件。 few shot背景: predictor based模型结构搜索, 需要采样足够多的模型结构来训练预测器，将模型结构训到较高的指标需要加很多trick并且需要训练非常久，从而限制采样子网络的数量。代理任务可以快速获得模型的精度，但是代理任务的精度分布与加入trick并且训练更久的精度分布之间会有diff。第二阶段的目标就是，基于第一阶段的代理任务采样的模型结构与模型精度之间的关联性，在只采样非常少量模型结构在非代理任务（加入trick并且训练更久）上的精度情况下，就可以准确的预测任意模型结构在非代理任务上的精度。

本demo基于paddleslim自研NAS算法[GP-NAS:Gaussian Process based Neural Architecture Search](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_GP-NAS_Gaussian_Process_Based_Neural_Architecture_Search_CVPR_2020_paper.pdf)（CVPR2020）
