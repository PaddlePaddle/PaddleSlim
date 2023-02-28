# QAT Constraints 原理

## 1. 概述
约束（Constraints）是指在模拟量化训练阶段，施加到模型上的一些约束规则。约束的对象包括且不限于：模型参数、量化参数、Layer的执行。
需要添加约束的原因：模拟量化的前向数值计算需要和推理的前向数值计算对齐。
实现独立且可扩展的约束模块的原因：不同的推理库和硬件，对量化推理的支持程度和方式不同，则在模拟量化训练阶段需要添加不同的约束。

## 2. 各种约束介绍


### 2.1 Conv/MatMul Bias约束
给定图1所示的计算图，FP32数值类型的Weight和Input经过矩阵乘，再加上一个同是FP32数值类型的Bias，最终的FP32数值类型的输出交给后续的Layer，这里以SkipLayernorm为例。

<center><img src="https://user-images.githubusercontent.com/7534971/221735515-24a31167-6dd0-43c2-a1ac-26e3bab86e42.png" width="300"></center>
<center>图 1 - Matmul Add计算示意图</center>

对于上述计算，可以列举出三种模拟 Int8 量化方式，分别对应三种 Int8 推理实现。

#### 第一种量化实现

这种实现无需特殊约束，只用正常的模拟量化训练。对该实现介绍的目的是供其它量化方式对比。
如下图所示，在模拟量化阶段，在 Matmul 的两个输入之后插入 Per-Tensor 的 QDQ Layer（量化反量化操作），来模拟对weight和input的量化。
weigh 和 input 对应的量化 scale 分别为 $S_w$ 和 $S_i$。右图为推理实现，其中：

-  删除 weight 的模拟量化 Layer，以 Int8 数值格式存储 weight
- 将 Input 的模拟量化 Layer 替换为量化 Layer。量化 Layer 负责将 FP32 类型的 input 量化为 Int8 数值类型，所用量化 scale 为模拟量化阶段统计到的 $S_i$
- Matmul以Int8数值类型进行乘计算，并将结果收集累加为Int32数值
- Matmul之后的Per-Tensor反量化Layer将Int32数值反量化为FP32，所用量化scale为$S_w * S_i$
- Add正常用FP32数值类型计算

<center><img src="https://user-images.githubusercontent.com/7534971/221735552-4c2df9e2-2a5f-4e25-a700-ad501be759f5.png" width="500"></center>

<center>图2 第一种量化实现示意图</center>

计算公式如下：

$$S_w = \frac{AbsMax_w }{2^{8-1}}$$
$$S_i = \frac{AbsMax_i}{2^{8-1}}$$
$$q\_weight = round(\frac{weight} { S_w})$$
$$q\_input = round(\frac{input} { S_i})$$
$$dq\_output = output * S_w * S_i $$

#### 第二种量化实现

这种实现需要对量化Scale进行约束：
1. Bias 的量化 scale 必须与 Matmul 的反量化 scale 一致
2. Matmul 的量化 scale 受 Bias 限制，需要保证将量化后的 Bias 限制在 16bits 以内（仅限TI芯片）
如下图左侧模型结构所示，不仅模拟量化Matmul两个输入，还模拟量化了Bias。对Bias的模拟量化与传统模拟量化不同，量化 scale 不是通过统计Bias收集得到的，而是直接用的 $S_w * S_i$，也就是 Matmul Layer 的反量化 scale。
这里就是对Matmul+Bias的一个约束规则，在模拟量化训练阶段，要以 $S_w * S_i$ 为量化 scal e对 Bias 进行模拟量化。该约束是为了与推理实现对齐，如下图右侧模型结构所示。
在推理实现中，Matmul 以 Int8 数值类型进行乘计算，收集相乘后的数值，累加到 Int32 类型的 accumulator 中。为了能将 Bias 也累加到这个 accumulator，需要将 Bias 量化为 Int32。并且，量化Bias 所用的量化 scale 需要与 accumulator 的量化 scale 一样，即 $S_w * S_i$。

<center><img src="https://user-images.githubusercontent.com/7534971/221735606-9820464d-797b-4437-bc7c-fbfc5bc3a0d3.png" width="500"></center>
<center>图3 第二种量化实现示意图</center>

整个推理执行的公式为：

$$fp32\_output = (q\_weight * q\_input + q\_bias) * S_w * S_i $$

其中：

- $q\_weight$: 量化后的weight
- $q\_input$：计算方式为 $round(\frac{fp32\_input}{S_i})$
- $q\_bias$: 量化后的bias，计算方式为 $round(\frac{fp32\_bias} {S_w * S_i})$

对于TI芯片，基本过程如上所述，但是在某些场景下，需要限制量化后的bias到Int16的范围内。用公式表示为：
$$\frac{abs\_max(fp32\_bias)} {S_w * S_i} <= 2^{15}$$

在模拟量化训练过程中，需要满足上述公式的约束，即本小节开始处提到的第二个约束。

#### 第三种量化实现

这种量化方式，只需对Add单独加约束，即：保证Add Layer的两个输入的量化scale保持一致。
如下图左侧模型结构所示，Matmul和Add相互独立，都分别添加了模拟量化Layer。Add的两个输入的量化scale为与Matmul输入的量化scale无关。
在推理时，无法直接将bias累加到matmul的Int32 accumulator上，因为bias的量化scale与accumulator不一样。而是，先用$S_w * S_i$将Int32 accumulator反量化为FP32，然后再用$S_o$将FP32数值量化为Int8，并与Int8数值类型的Bias相加。最终将Add结果用 $S_o$ 反量化为FP32数值。整个推理的计算公式为：

$$fp32\_output = (round(\frac{(q\_weight * q\_input) * S_w*S_i}{S_o}) + q\_bias) * S_o$$

其中：

$q\_bias$: 量化后的bias，计算方式为 $round(\frac{fp32\_bias} {S_o})$
这种实现方式，理论上没有第二种合理，这里列出来是为了对比第二种方法中的Bias的量化方式。

<center><img src="https://user-images.githubusercontent.com/7534971/221735648-f6fd3db2-1ec3-4253-a989-f68180169d6b.png" width="500"></center>
<center>图4 - 第三种量化实现示意图</center>


### 2.2 Convolution/mul BatchNorm约束

本小节介绍2种在模拟量化训练中处理convolution batchnorm的方式。
Batch normalization 被普遍用于计算机视觉模型训练，它可以减少模型层与层之间的影响，是模型训练更稳定和快速。
训练时，公式如下：

$$x_{bn} = \gamma (\frac{x-\mu_{B}}{\sigma_B} ) + \beta$$

推理时，公式如下：

$$x_{bn} = \gamma (\frac{x-\mu}{\sigma} ) + \beta$$

其中，$\mu_B$ 和 $\sigma_B$ 是当前当前单个 batch 的平均值和标准差。$\mu$ 和 $\sigma$ 是在训练阶段，从多个batch中统计得到平均值和标准差。
在推理时，会将batch normalizaiton融合到convolution或mul layer中。公式如下：

$$W_{inference} = \frac{\gamma * W}{\sigma}$$
$$Bias_{inference} = \beta - \frac{\gamma \mu}{\sigma}$$

在模拟量化训练阶段，需要模拟上述convolution和batch normalization的融合操作。

可以有两个选择：
- 策略1：Unfreeze BN, 使用单个batch的统计的平均值（$\mu_B$）和标准差($\sigma_B$)
  - 缺点：训练不稳定。batch间BN的统计值$\sigma_B$变化比较大，与BN融合之后，convolution的weight也会随$\sigma_B$频繁变化，最终导致weight量化scales的不稳定。而在推理时，convolution融合的是从全局统计的$\sigma$。所以，训练时weight量化scales的不稳定，会体现为推理精度的不稳定。
  - 现象：如图5绿色曲线。（Batch normalization without corrections (green) shows a lot of jitter dueto the changing scaling of weights from batch to batch.）
- 策略2：Freeze BN，即使用训练阶段统计的平均值($\mu$)和标准差($\sigma$)
  - 缺点：没有用到batch norm的特性，对激活做nomaliztion所用的mean和variance不符合当前batch的激活分布，导致量化训练不稳定
  - 现象：

在策略1中，训练不稳定的直接原因是『weight的量化scale频繁变化』，使用moving average来计算weight的scale，会缓解问题，但不能完全避免该问题，如图5橙色曲线所示。

<center><img src="https://user-images.githubusercontent.com/7534971/221735683-68bdde29-6b00-4943-98b1-ea8cda946dc2.png" width="500"></center>
<center>图5 - 各种Conv BN训练方法效果对比</center>


最优的方案是综合策略1和策略2的优点，让weight使用全局的BN参数$\sigma$，使weight更稳定。让激活使用当前batch统计到的$\sigma_B$，使激活BN整准确。该方案如图6所示

<center><img src="https://user-images.githubusercontent.com/7534971/221735696-f78fdaff-2067-4a76-bb92-c99ae4740f2f.png" width="500"></center>
<center>图6 - Conv BN量化训练矫正方案示意图</center>


图6方案有如下3个关键点：
1. 让conv weight更稳定
在量化之前，将全局统计的BN参数$\sigma$合并到conv weight，确保weight量化的稳定性。
2. 让激活BN更准确
在训练前期，通过乘上$\frac{\sigma}{\sigma_B}$来恢复weight融合对激活的影响，使激活是经过$\sigma_B$和$\mu_B$ normalize后的结果，行为更接近训练时标准的BN. 在上图中，对应Freeze Batch norm开关为0。
3. Freeze BN
经过一段unfreeze batch norm的训练，开启freeze batch norm开关，使激活的normalization也使用全局的BN参数，与推理保持一致。相当于，消除了BN带来的训练与推理的差异，专心学习量化。
上述方案，效果如图2.4.1中蓝色曲线所示。
