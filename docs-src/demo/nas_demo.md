# 网络结构搜索示例

本示例介绍如何使用网络结构搜索接口，搜索到一个更小或者精度更高的模型，该文档仅介绍paddleslim中SANAS的使用及如何利用SANAS得到模型结构，完整示例代码请参考sa_nas_mobilenetv2.py或者block_sa_nas_mobilenetv2.py。

## 接口介绍
请参考。

### 1. 配置搜索空间
详细的搜索空间配置可以参考<a href='../../../paddleslim/nas/nas_api.md'>神经网络搜索API文档</a>。
```
config = [('MobileNetV2Space')]

```

### 2. 利用搜索空间初始化SANAS实例
```
from paddleslim.nas import SANAS

sa_nas = SANAS(
    config,
    server_addr=("", 8881),
    init_temperature=10.24,
    reduce_rate=0.85,
    search_steps=300,
    is_server=True)

```

### 3. 根据实例化的NAS得到当前的网络结构
```
archs = sa_nas.next_archs()
```

### 4. 根据得到的网络结构和输入构造训练和测试program
```
import paddle.fluid as fluid

train_program = fluid.Program()
test_program = fluid.Program()
startup_program = fluid.Program()

with fluid.program_guard(train_program, startup_program):
    data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    for arch in archs:
        data = arch(data)
    output = fluid.layers.fc(data, 10)
    softmax_out = fluid.layers.softmax(input=output, use_cudnn=False)
    cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)

    test_program = train_program.clone(for_test=True)
    sgd = fluid.optimizer.SGD(learning_rate=1e-3)
    sgd.minimize(avg_cost)
    
```

### 5. 根据构造的训练program添加限制条件
```
from paddleslim.analysis import flops

if flops(train_program) > 321208544:
    continue
```

### 6. 回传score
```
sa_nas.reward(score)
```
