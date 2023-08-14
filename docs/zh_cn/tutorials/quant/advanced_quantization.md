# 量化策略详细教程
近年来，Transformer模型已在各个领域得到广泛采用，尤其是生成式语言大模型极大地推动了人工智能领域的发展。这些模型已经从数亿个参数发展到数千亿个参数，在有限的数据和 GPU 资源下运行，对于这些模型来说变得越来越具有挑战性。此时压缩技术变得格外重要，其中量化已成为减少内存占用和计算开销的通用和主要范例。然而，许多研究表明，Transformer模型往往会存在强烈的异常激活值，这使得它们难以量化。为了保持可接受的性能，这些异常值的存在要求激活具有更高的位宽或使用不同的数字格式、额外的微调或其他解决方法。本文档会介绍前沿的优化量化效果的几种策略，其中包括一些开源工作，也包括PaddleSlim自研的方法。

## **LLM量化效果Benchmark**

| 模型名称及大小 |  量化策略 | Finetuned下游任务 数据集nl2sql-指标acc | Pretrained开源任务 数据集C-eval-指标acc |
| --------- | ---------| ---------| ---------|
| LLama 13b | baseline-fp16 | 0.7832 | 0.3298 |
| - |  regular-int8 |  0.3405 | 0.2496 |
| - |  regular-smooth-int8 |  0.4236 | 0.2778 |
| - | regular-shift-int8 | 0.3335 | 0.2548 |
| - | regular-shift-smooth-int8 | 0.4526 | 0.2466 |
| - | paddleslim-int8 | **0.7807** | **0.3269** |
| - | regular-int4 | 0.7548 | 0.2927 |
| - | paddleslim-int4 | **0.7755** | **0.3232** |
| Bloom 7.1b |  baseline-fp16 | 0.7718 | 0.4108 |
| - |  regular-int8 |  0.7648 | 0.3826 |
| - |  regular-smooth-int8 |  0.7657 | 0.3870 |
| - | regular-shift-int8 | 0.6899 | 0.3997 |
| - | regular-shift-smooth-int8 | 0.7677 | 0.3722 |
| - | paddleslim-int8 | **0.7704** | **0.4063** |
| - | regular-int4 | 0.7518 | 0.3803 |
| - | paddleslim-int4 | **0.7682** | **0.3937** |
| ChatGLM2 6b |  baseline-fp16 | 0.7646 | 0.3157 |
| - |  regular-int8 |  0.6408 | 0.2994 |
| - |  regular-smooth-int8 |  0.6429 | 0.2986 |
| - | regular-shift-int8 | 0.5783 | 0.2726 |
| - | regular-shift-smooth-int8 | 0.5889 | 0.2429 |
| - | paddleslim-int8 | **0.7689** | **0.3455** |
| - | regular-int4 | 0.7420 | 0.2630 |
| - | paddleslim-int4 | **0.7566** | **0.3150** |


以下方法暂时仅支持Transformer模型，具体示例使用方法可参考[PaddleNLP LLM示例](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm#6-%E9%87%8F%E5%8C%96)，以下教程仅详细介绍API接口。

## 1. Shift功能

Shift算法来源于[Outlier Suppression+](https://arxiv.org/abs/2304.09145)。通过Layer Norm和Linear的bias进行数学等价的异常值缩放操作，有效将激活值的分布调整对称，有助于离线量化的精度提升。在PaddleSlim的实现中，对于前面有Layer Norm的Linear将使用数学等价的方式改变bias从而进行缩放操作，对于前面没有Layer Norm的Linear可以采用插入节点的方式实现，可通过参数shift_all_linears来控制是否需要shift前面没有Layer Norm的Linear。此外，PaddleSlim版本的Shift功能提供传入sample_function，如设置sample_function为None，Shift算法将完全对齐论文[Outlier Suppression+](https://arxiv.org/abs/2304.09145).

| **参数名**                   | **参数类型**                              | **参数释义**                              |
|-----------------------------|-----------------------------------------|-----------------------------------------|
| model | paddle.nn.Layer |必须传入的动态图模型 |
| model_config | dict | 必须传入的模型结构的配置 |
| shift_all_linears| bool | 可选参数，默认为False，若为True，则shift模型中全部Linear；若为False，则只shift模型中Layer Norm之后的Linear |
| sample_function | function | 可选参数，默认为None，若为None，采样方式为论文中相同方法，现可选的sample_function有MultiStepSampler和EMASampler，Shift时推荐使用EMASampler |

以下为简单的使用示例：
```shell
from paddleslim.quant.advanced import Shift, EMASampler

model = LLM()
model_config = {}
shift = Shift(model, model_config, sample_function=EMASampler())
for data in dataloader():
    model(data)
    shift.step += 1
shift.update_weight()
```



## 2. Smooth功能
Smooth算法来源于[SmoothQuant](https://arxiv.org/abs/2211.10438)。通过Layer Norm和Linear的weight和bias进行数学等价的异常值缩放操作，有效减少激活值中的异常值，有助于离线量化的精度提升。在PaddleSlim的实现中，与shift相同，对于前面有Layer Norm的Linear将使用数学等价的方式改变weight和bias从而进行缩放操作，对于前面没有Layer Norm的Linear可以采用插入节点的方式实现，可通过参数smooth_all_linears来控制是否需要smooth前面没有Layer Norm的Linear。此外，PaddleSlim版本的Smooth功能还提供搜索功能，搜索功能文档见下文。

| **参数名**                   | **参数类型**                              | **参数释义**                              |
|-----------------------------|-----------------------------------------|-----------------------------------------|
| model | paddle.nn.Layer |必须传入的动态图模型 |
| model_config | dict | 必须传入的模型结构的配置 |
| alpha | float | 可选参数，默认为0.5 |
| smooth_all_linears| bool | 可选参数，默认为False，若为True，则shift模型中全部Linear；若为False，则只shift模型中Layer Norm之后的Linear |
| sample_function | function | 可选参数，默认为None，若为None，采样方式为单batch数据，现可选的sample_function有MultiStepSampler和EMASampler，Smooth时推荐使用MultiStepSampler |
| search_function | function | 可选参数，默认为None，若为None，则不进行搜索，smooth方法与原论文保持一致 |

以下为简单的使用示例：
```shell
from paddleslim.quant.advanced import Smooth，MultiStepSampler

model = LLM()
model_config = {}
smooth = Smooth(model, model_config, sample_function=MultiStepSampler())
for data in dataloader():
    model(data)
    smooth.step += 1
smooth.update_weight()
```


注意：模型shift和smooth前后从理论数学角度应该是等价的，但从工程角度，输出具体数值可能会有稍微不同，所以模型shift/smooth前后的精度是大约一致的。如果精度出现很大差距，说明模型未解析成功。参数中`model_config`请按照模型的实际情况填写，此输入会影响模型结构解析，若结构解析出错，会导致模型精度不对，其中包含字段：
- `fused_qkv`：该模型是否融合了QKV，默认为True
- `linear_flag`：该模型判断Linear的名字字段，默认为`linear`
- `norm_flag`：该模型判断Layer Norm的名字字段，默认为`norm`
- `parallel_ffn`：该模型是否含有并行的FFN，默认为False
- `skip_norm_list`：该模型中需要被忽略的Layer Norm的名字字段，默认为空list

若模型中含有PostLayerNorm Shurtcut结构，则不支持对该模型进行smooth和shift。比如PaddleNLP中[ChatGLM结构](https://github.com/PaddlePaddle/PaddleNLP/blob/64f97979a62fba6a35a1177850cc22dbc91fade0/paddlenlp/transformers/chatglm/modeling.py#L360)存在PostLayerNorm Shurtcut结构，所以不支持对该模型进行shift/smooth。


## 3. PieceWiseSearch功能
根据[SmoothQuant](https://arxiv.org/abs/2211.10438)算法，的确能够有效减少异常值，但我们在大量的实验中发现，在某些情况下，比如权重值较大，尤其是权重和激活的异常值在同一通道时，直接根据SmoothQuant计算smooth scale的公式会导致权重值难量化的情况。并且，对于一个激活值，当异常值较多、数值范围较大，使用同一个alpha去smooth整个激活张量也并不合理。因此，PaddleSlim提出分段搜索功能，根据数值大小将激活分成K段，对于每一段进行alhpa和scale的搜索。

| **参数名**                   | **参数类型**                              | **参数释义**                              |
|-----------------------------|-----------------------------------------|-----------------------------------------|
| k_piece | int | 可选参数，分段数量，默认为1，1代表不分段 |
| bits_length | int | 可选参数，量化比特数，默认为8 |
| search_piece | bool | 可选参数，是否搜索分段数k，默认为False，若为True，将会从1到k搜索合适的k |
| search_alpha_min | float | 可选参数，搜索alpha最小值，默认为0.2 |
| search_alpha_max | float | 可选参数，搜索alpha最大值，默认为0.8 |
| search_scale_min | float | 可选参数，搜索scale最小值，默认为1. |
| search_scale_max | float | 可选参数，搜索scale最大值，默认为1. |
| weight_quant_method | str | 可选参数，权重量化方法，可选`abs_max`，`abs_max_channel_wise`，`avg`，默认为`abs_max_channel_wise`  |
| act_quant_method | str | 可选参数，激活量化方法，可选`abs_max`，`avg`，默认为`abs_max` |
| loss_function | function | 可选参数，搜索时使用的误差函数，默认为mse_loss |


```shell
from paddleslim.quant.advanced import Smooth, MultiStepSampler, PieceWiseSearch, mse_loss

search_func =PieceWiseSearch(
                k_piece=3,
                bits_length=8,
                search_piece=False,
                search_alpha_min=0.2,
                search_alpha_max=0.8,
                search_scale_min=1.,
                search_scale_max=5.,
                weight_quant_method='abs_max_channel_wise',
                act_quant_method='abs_max',
                loss_function=mse_loss
            )
model = LLM()
model_config = {}
smooth = Smooth(model, model_config, sample_function=MultiStepSampler(), search_function=search_func)
for data in dataloader():
    model(data)
    smooth.step += 1
smooth.update_weight()

```

## 4. GPTQ
GPTQ算法来自[GPTQ](https://arxiv.org/abs/2210.17323)，该算法逐步按照行量化权重,利用海森矩阵来不断更新未量化的权重，在低比特Weight Only Int4量化表现良好。GPTQ默认使用搭配使用[RPTQ](https://arxiv.org/abs/2304.01089)，若不想搭配RPTQ，调用fasterquant时设置act_order=False即可。

| **参数名**                   | **参数类型**                              | **参数释义**                              |
|-----------------------------|-----------------------------------------|-----------------------------------------|
| layer | paddle.nn.Layer |必须入的需要量化的层，现仅支持nn.Linear，ColumnParallelLinear和RowParallelLinear类型 |
| model_config | dict | 必须传入的模型结构的配置 |
| quant_bits| int | 可选参数，量化比特数，默认为4 |
| weight_quant_method | str | 可选参数，权重量化方法，可选`abs_max`，`abs_max_channel_wise`，`avg`，默认为`abs_max_channel_wise` |

```shell
from paddleslim.quant.advanced import GPTQ

model = LLM()
for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == paddle.nn.Linear:
        gptq_layer = GPTQ(cur_layer)
        # sample data
        for data in dataloader():
            model(data)
        # quant weight
        gptq_layer.fasterquant(act_order=True)
```


## 5. LayerWiseQuantError
LayerWiseQuantError是按层级别分析量化损失的方法，对于模型中每一层，量化后，计算当前层量化输出和原始模型输出的误差。
| **参数名**                   | **参数类型**                              | **参数释义**                              |
|-----------------------------|-----------------------------------------|-----------------------------------------|
| layer | paddle.nn.Layer |必须入的需要量化的层，现仅支持nn.Linear，ColumnParallelLinear和RowParallelLinear类型 |
| weight_bits | int | 可选参数，权重量化比特数，默认为8 |
| act_bits| int | 可选参数，激活量化比特数，默认为8 |
| weight_quant_method| str | 可选参数，权重量化方法，可选`abs_max`，`abs_max_channel_wise`，`avg`，默认为`abs_max_channel_wise` |
| act_quant_method| str | 可选参数，激活量化方法，可选`abs_max`，`avg` |
| loss_function | function | 可选参数，使用的误差函数，默认为mse_loss |

```shell
from paddleslim.quant.advanced import LayerWiseQuantError

model = LLM()
for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == paddle.nn.Linear:
        gptq_layer = LayerWiseQuantError(cur_layer)

for data in dataloader():
    model(data)

for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == LayerWiseQuantError:
        print(cur_name, cur_layer.losses.mean())
```
