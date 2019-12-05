# paddleslim.quant API文档

## 量化训练API

### 量化配置
```
quant_config_default = {
    'weight_quantize_type': 'abs_max',
    'activation_quantize_type': 'abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    # ops of name_scope in not_quant_pattern list, will not be quantized
    'not_quant_pattern': ['skip_quant'],
    # ops of type in quantize_op_types, will be quantized
    'quantize_op_types':
    ['conv2d', 'depthwise_conv2d', 'mul', 'elementwise_add', 'pool2d'],
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. defaulf is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # if set quant_weight_only True, then only quantize parameters of layers which need to be quantized,
    # and activations will not be quantized.
    'quant_weight_only': False
}
```
功能：设置量化训练需要的配置。

参数说明：
- ``weight_quantize_type(str)``: 参数量化方式。可选``'abs_max'``,  ``'channel_wise_abs_max'``, ``'range_abs_max'``, ``'moving_average_abs_max'``。 默认``'abs_max'``。
- ``activation_quantize_type(str)``: 激活量化方式，可选``'abs_max'``, ``'range_abs_max'``, ``'moving_average_abs_max'``，默认``'abs_max'``。
- ``weight_bits(int)``: 参数量化bit数，默认8, 推荐设为8。
- ``activation_bits(int)``: 激活量化bit数，默认8， 推荐设为8。
- ``not_quant_pattern(str or str list)``: 所有``name_scope``包含``'not_quant_pattern'``字符串的``op``，都不量化, 设置方式请参考``fluid.name_scope()``。
- ``quantize_op_types(str of list)`` : 需要进行量化的``op``类型，目前支持``'conv2d', 'depthwise_conv2d', 'mul' ``。
- ``dtype(int8)``: 量化后的参数类型，默认 ``int8``, 目前仅支持``int8``。
- ``window_size(int)``: ``'range_abs_max'``量化方式的``window size``，默认10000。
- ``moving_rate(int)``: ``'moving_average_abs_max'``量化方式的衰减系数，默认 0.9。
- ``quant_weight_only(bool)`` : 是否只量化参数，如果设为``True``，则激活不进行量化，默认``False``。目前暂不支持设置为``True``。 设置为``True``时，只量化参数，这种方式不能减少显存占用和加速，只能用来减少带宽。


### paddleslim.quant.quant_aware(program, place, config, scope=None, for_test=False)
功能：在``program``中加入量化和反量化``op``, 用于量化训练。具体如图所示：


**参数介绍：**
* ``program (fluid.Program)``: 传入训练或测试``program``。
* ``place(fluid.CPUPlace or fluid.CUDAPlace)``: 该参数表示``Executor``执行所在的设备。
* ``config(dict)``: 量化配置表。
* ``scope(fluid.Scope, optional)``: 传入用于存储``Variable``的``scope``，需要传入``program``所使用的``scope``，一般情况下，是``fluid.global_scope()``。设置为``None``时将使用``fluid.global_scope()``，默认值为``None``。
* ``for_test(bool)``: 如果``program``参数是一个测试``program``，``for_test``应设为``True``，否则设为``False``。

**返回值：**

含有量化和反量化``operator``的``program``
* 当``for_test=False``，返回类型为``fluid.CompiledProgram``， **注意，此返回值不能用于保存参数**。
* 当``for_test=True``，返回类型为``fluid.Program``。

**注意事项**:
* 此接口会改变``program``结构，并且可能增加一些``persistable``的变量，所以加载模型参数时请注意和相应的``program``对应。
* 此接口底层经历了``fluid.Program``-> ``fluid.framework.IrGraph``->``fluid.Program``的转变，在``fluid.framework.IrGraph``中没有``Parameter``的概念，``Variable``只有``persistable``和``not persistable``的区别，所以在保存和加载参数时，请使用``fluid.io.save_persistables``和``fluid.io.load_persistables``接口。
* 由于此接口会根据``program``的结构和量化配置来对``program``添加op，所以``Paddle``中一些通过``fuse op``来加速训练的策略不能使用。已知以下策略在使用量化时必须设为``False``： ``fuse_all_reduce_ops, sync_batch_norm``。
* 如果传入的``program``中存在和任何op都没有连接的``Variable``，则会在量化的过程中被优化掉。



### paddleslim.quant.convert(program, place, config, scope=None, save_int8=False)


功能：把训练好的量化``program``，转换为可用于保存``inference model``的``program``。

**参数介绍：**
- ``program (fluid.Program)``: 传入测试``program``。
- ``place(fluid.CPUPlace or fluid.CUDAPlace)``: 该参数表示``Executor``执行所在的设备。
- ``config(dict)``: 量化配置表。
- ``scope(fluid.Scope)``: 传入用于存储``Variable``的``scope``，需要传入``program``所使用的``scope``，一般情况下，是``fluid.global_scope()``。设置为``None``时将使用``fluid.global_scope()``，默认值为``None``。
- ``save_int8（bool）``: 是否需要返回参数为``int8``的``program``。该功能目前只能用于确认模型大小。默认值为``False``。

返回值：
- ``program (fluid.Program)``: freezed program，可用于保存inference model，参数为``float32``类型，但其数值范围可用int8表示。
- ``int8_program (fluid.Program)``: freezed program，可用于保存inference model，参数为``int8``类型。当``save_int8``为``False``时，不返回该值。

**注意事项**:
* 因为该接口会对``op``和``Variable``做相应的删除和修改，所以此接口只能在训练完成之后调用。如果想转化训练的中间模型，可加载相应的参数之后再使用此接口。

**使用示例**

```python
#encoding=utf8
import paddle.fluid as fluid
import paddleslim.quant as quant


train_program = fluid.Program()

with fluid.program_guard(train_program):
    image = fluid.data(name='x', shape=[None, 1, 28, 28])
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    conv = fluid.layers.conv2d(image, 32, 1)
    feat = fluid.layers.fc(conv, 10, act='softmax')
    cost = fluid.layers.cross_entropy(input=feat, label=label)
    avg_cost = fluid.layers.mean(x=cost)

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
eval_program = train_program.clone(for_test=True)
#配置
config = {'weight_quantize_type': 'abs_max',
        'activation_quantize_type': 'moving_average_abs_max'}
build_strategy = fluid.BuildStrategy()
exec_strategy = fluid.ExecutionStrategy()
#调用api
quant_train_program = quant.quant_aware(train_program, place, config, for_test=False)
quant_eval_program = quant.quant_aware(eval_program, place, config, for_test=True)
#关闭策略
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
quant_train_program = quant_train_program.with_data_parallel(
    loss_name=avg_cost.name,
    build_strategy=build_strategy,
    exec_strategy=exec_strategy)

inference_prog = quant.convert(quant_eval_program, place, config)
```

更详细的用法请参考 <a href='../../demo/quant/quant_aware/README.md'>量化训练demo</a>。

## 离线量化API
```
paddleslim.quant.quant_post(executor,
           model_dir,
           quantize_model_path,
           sample_generator,
           model_filename=None,
           params_filename=None,
           batch_size=16,
           batch_nums=None,
           scope=None,
           algo='KL',
           quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"])

```
**参数介绍：**
- ``executor (fluid.Executor)``: 执行模型的executor，可以在cpu或者gpu上执行。
- ``model_dir（str)``: 需要量化的模型所在的文件夹。
- ``quantize_model_path(str)``: 保存量化后的模型的路径
- ``sample_generator(python generator)``: 读取数据样本，每次返回一个样本。
- ``model_filename(str, optional)``: 模型文件名，如果需要量化的模型的参数存在一个文件中，则需要设置``model_filename``为模型文件的名称，否则设置为``None``即可。默认值是``None``。
- ``params_filename(str)``: 参数文件名，如果需要量化的模型的参数存在一个文件中，则需要设置``params_filename``为参数文件的名称，否则设置为``None``即可。默认值是``None``。
- ``batch_size(int)``: 每个batch的图片数量。默认值为16 。
- ``batch_nums(int, optional)``: 迭代次数。如果设置为``None``，则会一直运行到``sample_generator`` 迭代结束， 否则，迭代次数为``batch_nums``, 也就是说参与对``Scale``进行校正的样本个数为 ``'batch_nums' * 'batch_size' ``.
- ``scope(fluid.Scope, optional)``: 用来获取和写入``Variable``, 如果设置为``None``,则使用``fluid.global_scope()``. 默认值是``None``.
- ``algo(str)``: 量化时使用的算法名称，可为``'KL'``或者``'direct'``。该参数仅针对激活值的量化，因为参数值的量化使用的方式为``'channel_wise_abs_max'``. 当``algo`` 设置为``'direct'``时，使用校正数据的激活值的绝对值的最大值当作``Scale``值，当设置为``'KL'``时，则使用``KL``散度的方法来计算``Scale``值。默认值为``'KL'``。
- ``quantizable_op_type(list[str])``: 需要量化的``op``类型列表。默认值为``["conv2d", "depthwise_conv2d", "mul"]``。

**返回值**

无。

**注意事项**
因为该接口会收集校正数据的所有的激活值，所以使用的校正图片不能太多。``'KL'``散度的计算也比较耗时。

**使用示例**

> 注： 此示例不能直接运行，因为需要加载``${model_dir}``下的模型，所以不能直接运行。

```python
import paddle.fluid as fluid
import paddle.dataset.mnist as reader
from paddleslim.quant import quant_post
val_reader = reader.train()
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

exe = fluid.Executor(place)
quant_post(
        executor=exe,
        model_dir='./model_path',
        quantize_model_path='./save_path',
        sample_generator=val_reader,
        model_filename='__model__',
        params_filename='__params__',
        batch_size=16,
        batch_nums=10)
```
更详细的用法请参考 <a href='../../demo/quant/quant_post/README.md'>离线量化demo</a>。

## Embedding 量化API
```
paddleslim.quant.quant_embedding(program, place, config, scope=None)
```
**参数介绍:**
- ``program(fluid.Program)`` : 需要量化的program
- ``scope(fluid.Scope, optional)``: 用来获取和写入``Variable``, 如果设置为``None``,则使用``fluid.global_scope()``.
- ``place(fluid.CPUPlace or fluid.CUDAPlace)``: 运行program的设备
- ``config(dict)`` : 定义量化的配置。可以配置的参数有：
    - ``'params_name'`` (str, required): 需要进行量化的参数名称，此参数必须设置。
    - ``'quantize_type'`` (str, optional): 量化的类型，目前支持的类型是``'abs_max'``, 待支持的类型有 ``'log', 'product_quantization'``。 默认值是``'abs_max'``.
    - ``'quantize_bits'``（int, optional): 量化的``bit``数，目前支持的``bit``数为8。默认值是8.
    - ``'dtype'``(str, optional): 量化之后的数据类型， 目前支持的是``'int8'``. 默认值是``int8``。
    - ``'threshold'``(float, optional): 量化之前将根据此阈值对需要量化的参数值进行``clip``. 如果不设置，则跳过``clip``过程直接量化。

**返回值**

量化之后的program，类型为``fluid.Program``

**使用示例**
```python
import paddle.fluid as fluid
import paddleslim.quant as quant

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

config = {'params_name': 'emb', 'quantize_type': 'abs_max'}
quant_program = quant.quant_embedding(infer_program, place, config)
```

更详细的用法请参考 <a href='../../demo/quant/quant_embedding/README.md'>Embedding量化demo</a>。
