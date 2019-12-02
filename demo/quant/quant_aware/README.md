# 在线量化示例

本示例介绍如何使用在线量化接口，来对训练好的分类模型进行量化, 可以减少模型的存储空间和显存占用。

## 接口介绍

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

量化配置表。
参数说明：
- weight_quantize_type(str): 参数量化方式。可选'abs_max',  'channel_wise_abs_max', 'range_abs_max', 'moving_average_abs_max'，默认'abs_max'。
- activation_quantize_type(str): 激活量化方式，可选'abs_max', 'range_abs_max', 'moving_average_abs_max'，默认'abs_max'。
- weight_bits(int): 参数量化bit数，默认8。
- activation_bits(int): 激活量化bit数，默认8。
- not_quant_pattern(str or str list): 所有name_scope包含not_quant_pattern字符串的op，都不量化。
- quantize_op_types(str of list): 需要进行量化的op类型，目前支持'conv2d', 'depthwise_conv2d', 'mul', 'elementwise_add', 'pool2d'。
- dtype(int8): 量化后的参数类型，默认int8。
- window_size(int): 'range_abs_max'量化的window size，默认10000。
- moving_rate(int): moving_average_abs_max 量化的衰减系数，默认 0.9。
- quant_weight_only(bool): 是否只量化参数，如果设为True，则激活不进行量化，默认False。

```
def quant_aware(program, 
                place, 
                config,
                scope=None, 
                for_test=False)
```

该接口会对传入的program插入可训练量化op。
参数介绍：
- program (fluid.program): 传入训练或测试program。
- place(fluid.CPUPlace or fluid.CUDAPlace): 该参数表示Executor执行所在的设备。
- config(dict): 量化配置表。
- scope(fluid.Scope): 传入用于存储var的scope，需要传入program所使用的scope，一般情况下，是fluid.global_scope()。
- for_test(bool): 如果program参数是一个测试用program，for_test应设为True，否则设为False。

返回参数：
-  program(fluid.Program): 插入量化op后的program。
   注意：如果for_test为False，这里返回的program是compiled program。

```
def convert(program, 
            place, 
            config, 
            scope=None, 
            save_int8=False)
```

把训练好的量化program，转换为可用于保存inference model的program。
注意，本接口返回的program，不可用于训练。
参数介绍：
- program (fluid.program): 传入测试program。
- place(fluid.CPUPlace or fluid.CUDAPlace): 该参数表示Executor执行所在的设备。
- config(dict): 量化配置表。
- scope(fluid.Scope): 传入用于存储var的scope，需要传入program所使用的scope，一般情况下，是fluid.global_scope()。
- save_int8（bool）: 是否需要导出参数为int8的program。(该功能目前只能用于确认模型大小)

返回参数：
- program (fluid.program): freezed program，可用于保存inference model，参数为float32类型，但其数值范围可用int8表示。
- int8_program (fluid.program): freezed program，可用于保存inference model，参数为int8类型。


## 分类模型的离线量化流程

### 1. 配置量化参数

```
quant_config = {
    'weight_quantize_type': 'abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'not_quant_pattern': ['skip_quant'],
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9,
    'quant_weight_only': False
}
```

### 2. 对训练和测试program插入可训练量化op

```
val_program = quant_aware(val_program, place, quant_config, scope=None, for_test=True)

compiled_train_prog = quant_aware(train_prog, place, quant_config, scope=None, for_test=False)
```

### 3.关掉指定build策略

```
build_strategy = fluid.BuildStrategy()
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
exec_strategy = fluid.ExecutionStrategy()
compiled_train_prog = compiled_train_prog.with_data_parallel(
        loss_name=avg_cost.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
```

### 4. freeze program

```
float_program, int8_program = convert(val_program, 
                                      place,
                                      quant_config,
                                      scope=None,
                                      save_int8=True)
```

### 5.保存预测模型

```
fluid.io.save_inference_model(
    dirname=float_path,
    feeded_var_names=[image.name],
    target_vars=[out], executor=exe,
    main_program=float_program,
    model_filename=float_path + '/model',
    params_filename=float_path + '/params')

fluid.io.save_inference_model(
    dirname=int8_path,
    feeded_var_names=[image.name],
    target_vars=[out], executor=exe,
    main_program=int8_program,
    model_filename=int8_path + '/model',
    params_filename=int8_path + '/params')
```




