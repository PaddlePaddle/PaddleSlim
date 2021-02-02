# Embedding量化

Embedding量化将网络中的Embedding参数从`float32`类型量化到 `8-bit`整数类型，在几乎不损失模型精度的情况下减少模型的存储空间和显存占用。

Embedding量化仅能减少模型参数的体积，并不能显著提升模型预测速度。
## 使用方法

在预测时调用paddleslim `quant_embedding`接口，主要实现代码如下：

```python
import paddleslim
place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
exe = paddle.static.Executor(place)
main_program = paddleslim.quant.quant_embedding(main_program, place, config)
```

详细代码与例程请参考：[Embedding量化](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_embedding)
