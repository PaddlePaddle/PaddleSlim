# QAT Constraints 原理

## 1. 概述
约束（Constraints）是指在模拟量化训练阶段，施加到模型上的一些约束规则。约束的对象包括且不限于：模型参数、量化参数、Layer 的执行。
需要添加约束的原因：模拟量化的前向数值计算需要和推理的前向数值计算对齐。
实现独立且可扩展的约束模块的原因：不同的推理库和硬件，对量化推理的支持程度和方式不同，则在模拟量化训练阶段需要添加不同的约束。

## 2. 各种约束介绍


### 2.1 Conv/MatMul Bias约束
给定图1所示的计算图，FP32 数值类型的 Weight 和 Input 经过矩阵乘，再加上一个同是 FP32 数值类型的 Bias，最终的 FP32 数值类型的输出交给后续的 Layer，这里以 SkipLayernorm 为例。

<div align="center"><img src="https://user-images.githubusercontent.com/7534971/221735515-24a31167-6dd0-43c2-a1ac-26e3bab86e42.png" width="300"></div>
<div align="center">图 1 - Matmul Add计算示意图</div>

对于上述计算，可以列举出三种模拟 Int8 量化方式，分别对应三种 Int8 推理实现。

#### 第一种量化实现

这种实现无需特殊约束，只用正常的模拟量化训练。对该实现介绍的目的是供其它量化方式对比。
如下图所示，在模拟量化阶段，在 Matmul 的两个输入之后插入 Per-Tensor 的 QDQ Layer（量化反量化操作），来模拟对weight和input的量化。
weigh 和 input 对应的量化 scale 分别为 $S_w$ 和 $S_i$。右图为推理实现，其中：

-  删除 weight 的模拟量化 Layer，以 Int8 数值格式存储 weight
- 将 Input 的模拟量化 Layer 替换为量化 Layer。量化 Layer 负责将 FP32 类型的 input 量化为 Int8 数值类型，所用量化 scale 为模拟量化阶段统计到的 $S_i$
- Matmul 以 Int8 数值类型进行乘计算，并将结果收集累加为 Int32 数值
- Matmul 之后的 Per-Tensor 反量化 Layer 将 Int32 数值反量化为FP32，所用量化 scale为$S_w * S_i$
- Add 正常用 FP32 数值类型计算

<div align="center"><img src="https://user-images.githubusercontent.com/7534971/221735552-4c2df9e2-2a5f-4e25-a700-ad501be759f5.png" width="500"></div>

<div align="center">图2 第一种量化实现示意图</div>

计算公式如下：

$$S_w = \frac{AbsMax_w }{2^{8-1}}$$
$$S_i = \frac{AbsMax_i}{2^{8-1}}$$
$$QWeight = round(\frac{weight} { S_w})$$
$$QInput = round(\frac{input} { S_i})$$
$$DqOutput = output * S_w * S_i $$

#### 第二种量化实现

这种实现需要对量化 Scale 进行约束：
1. Bias 的量化 scale 必须与 Matmul 的反量化 scale 一致
2. Matmul 的量化 scale 受 Bias 限制，需要保证将量化后的 Bias 限制在 16bits 以内（仅限 TI 芯片）
如下图左侧模型结构所示，不仅模拟量化 Matmul 两个输入，还模拟量化了 Bias。对 Bias 的模拟量化与传统模拟量化不同，量化 scale 不是通过统计Bias收集得到的，而是直接用的 $S_w * S_i$，也就是 Matmul Layer 的反量化 scale。
这里就是对 Matmul+Bias 的一个约束规则，在模拟量化训练阶段，要以 $S_w * S_i$ 为量化 scale对 Bias 进行模拟量化。该约束是为了与推理实现对齐，如下图右侧模型结构所示。
在推理实现中，Matmul 以 Int8 数值类型进行乘计算，收集相乘后的数值，累加到 Int32 类型的 accumulator 中。为了能将 Bias 也累加到这个 accumulator，需要将 Bias 量化为 Int32。并且，量化Bias 所用的量化 scale 需要与 accumulator 的量化 scale 一样，即 $S_w * S_i$。

<div align="center"><img src="https://user-images.githubusercontent.com/7534971/221735606-9820464d-797b-4437-bc7c-fbfc5bc3a0d3.png" width="500"></div>
<div align="center">图3 第二种量化实现示意图</div>

整个推理执行的公式为：

$$FP32Output = (QWeight * QInput + QBias) * S_w * S_i $$

其中：

- $QWeight$: 量化后的 weight
- $QInput$：计算方式为 $round(\frac{FP32Input}{S_i})$
- $QBias$: 量化后的 bias，计算方式为 $round(\frac{FP32Bias} {S_w * S_i})$

对于TI芯片，基本过程如上所述，但是在某些场景下，需要限制量化后的 bias 到 Int16 的范围内。用公式表示为：

$$\frac{AbsMax(FP32Bias)}{S_w * S_i} <= 2^{15}$$

在模拟量化训练过程中，需要满足上述公式的约束，即本小节开始处提到的第二个约束。

#### 第三种量化实现

这种量化方式，只需对 Add 单独加约束，即：保证 Add Layer 的两个输入的量化 scale 保持一致。
如下图左侧模型结构所示，Matmul 和 Add 相互独立，都分别添加了模拟量化 Layer。Add 的两个输入的量化 scale 为与 Matmul 输入的量化 scale 无关。
在推理时，无法直接将 bias 累加到 matmul 的 Int32 accumulator 上，因为 bias 的量化 scale 与 accumulator 不一样。而是，先用 $S_w * S_i$ 将 Int32 accumulator 反量化为 FP32，然后再用 $S_o$ 将 FP32 数值量化为 Int8，并与 Int8 数值类型的 Bias 相加。最终将 Add 结果用 $S_o$ 反量化为 FP32 数值。整个推理的计算公式为：

$$FP32Output = (round(\frac{(QWeight * QInput) * S_w*S_i}{S_o}) + QBias) * S_o$$

其中：

$QBias$: 量化后的 bias，计算方式为 $round(\frac{FP32Bias} {S_o})$
这种实现方式，理论上没有第二种合理，这里列出来是为了对比第二种方法中的 Bias 的量化方式。

<div align="center"><img src="https://user-images.githubusercontent.com/7534971/221735648-f6fd3db2-1ec3-4253-a989-f68180169d6b.png" width="500"></div>
<div align="center">图4 - 第三种量化实现示意图</div>


### 2.2 Convolution/mul BatchNorm约束

本小节介绍2种在模拟量化训练中处理 convolution batchnorm 的方式。
Batch normalization 被普遍用于计算机视觉模型训练，它可以减少模型层与层之间的影响，是模型训练更稳定和快速。
训练时，公式如下：

$$x_{bn} = \gamma (\frac{x-\mu_{B}}{\sigma_B} ) + \beta$$

推理时，公式如下：

$$x_{bn} = \gamma (\frac{x-\mu}{\sigma} ) + \beta$$

其中，$\mu_B$ 和 $\sigma_B$ 是当前当前单个 batch 的平均值和标准差。$\mu$ 和 $\sigma$ 是在训练阶段，从多个 batch 中统计得到平均值和标准差。
在推理时，会将 batch normalizaiton 融合到 convolution 或 mul layer 中。公式如下：

$$W_{inference} = \frac{\gamma * W}{\sigma}$$
$$Bias_{inference} = \beta - \frac{\gamma \mu}{\sigma}$$

在模拟量化训练阶段，需要模拟上述 convolution 和 batch normalization 的融合操作。

可以有两个选择：
- 策略1：Unfreeze BN, 使用单个 batch 的统计的平均值（$\mu_B$）和标准差($\sigma_B$)
  - 缺点：训练不稳定。batch 间 BN 的统计值 $\sigma_B$ 变化比较大，与 BN 融合之后，convolution 的 weight 也会随 $\sigma_B$ 频繁变化，最终导致 weight 量化 scales 的不稳定。而在推理时，convolution 融合的是从全局统计的 $\sigma$。所以，训练时 weight 量化 scales 的不稳定，会体现为推理精度的不稳定。
  - 现象：如图5绿色曲线。
- 策略2：Freeze BN，即使用训练阶段统计的平均值 $\mu$ 和标准差 $\sigma$
  - 缺点：没有用到 batch norm 的特性，对激活做 nomaliztion 所用的 mean 和 variance 不符合当前 batch 的激活分布，导致量化训练不稳定
  - 现象：

在策略1中，训练不稳定的直接原因是『weight的量化scale频繁变化』，使用 moving average 来计算 weight 的 scale，会缓解问题，但不能完全避免该问题，如图5橙色曲线所示。

<div align="center"><img src="https://user-images.githubusercontent.com/7534971/221735683-68bdde29-6b00-4943-98b1-ea8cda946dc2.png" width="500"></div>
<div align="center">图5 - 各种 Conv BN 训练方法效果对比</div>


最优的方案是综合策略1和策略2的优点，让 weight 使用全局的BN参数 $\sigma$，使 weight 更稳定。让激活使用当前 batch 统计到的 $\sigma_B$，使激活 BN 整准确。该方案如图6所示

<div align="center"><img src="https://user-images.githubusercontent.com/7534971/221735696-f78fdaff-2067-4a76-bb92-c99ae4740f2f.png" width="500"></div>
<div align="center">图6 - Conv BN 量化训练矫正方案示意图</div>


图6方案有如下3个关键点：
1. 让 conv weight 更稳定
在量化之前，将全局统计的 BN 参数 $\sigma$ 合并到 conv weight，确保 weight 量化的稳定性。
2. 让激活BN更准确
在训练前期，通过乘上 $\frac{\sigma}{\sigma_B}$ 来恢复 weight 融合对激活的影响，使激活是经过$\sigma_B$ 和 $\mu_B$ normalize 后的结果，行为更接近训练时标准的BN. 在上图中，对应 freeze Batch norm 开关为0。
3. Freeze BN
经过一段 unfreeze batch norm 的训练，开启 freeze batch norm 开关，使激活的 normalization 也使用全局的 BN 参数，与推理保持一致。相当于，消除了 BN 带来的训练与推理的差异，专心学习量化。
上述方案，效果如图6中蓝色曲线所示。
