# Embedding量化

Embedding量化将网络中的Embedding参数从`float32`类型量化到 `8-bit`或者 `16-bit` 整数类型，在几乎不损失模型精度的情况下减少模型的存储空间和显存占用。

Embedding量化仅能减少模型参数的体积，加快加载Embedding参数的速度，并不能显著提升模型预测速度。

## 使用方法

在预测时调用paddleslim `quant_embedding`接口，主要实现代码如下：

```python
import paddle
import paddle.fluid as fluid
import paddleslim.quant as quant
paddle.enable_static()
train_program = fluid.Program()
with fluid.program_guard(train_program):
    input_word = fluid.data(name="input_word", shape=[None, 1], dtype='int64')
    input_emb = fluid.embedding(
        input=input_word,
        is_sparse=False,
        size=[100, 128],
        param_attr=fluid.ParamAttr(name='emb',
        initializer=fluid.initializer.Uniform(-0.005, 0.005)))

infer_program = train_program.clone(for_test=True)

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 量化为8比特，Embedding参数的体积减小4倍，精度有轻微损失
config = {
         'quantize_op_types': ['lookup_table'],
         'lookup_table': {
             'quantize_type': 'abs_max',
             'quantize_bits': 8,
             'dtype': 'int8'
             }
         }

'''
# 量化为16比特，Embedding参数的体积减小2倍，精度损失很小
config = {
         'quantize_op_types': ['lookup_table'],
         'lookup_table': {
             'quantize_type': 'abs_max',
             'quantize_bits': 16,
             'dtype': 'int16'
             }
         }
'''

quant_program = quant.quant_embedding(infer_program, place, config)
```

详细代码与例程请参考：[Embedding量化](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_embedding)
