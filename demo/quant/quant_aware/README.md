# 在线量化示例

本示例介绍如何使用在线量化接口，来对训练好的分类模型进行量化, 可以减少模型的存储空间和显存占用。

## 接口介绍

请参考 <a href='../../../paddleslim/quant/quantization_api_doc.md'>量化API文档</a>。

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
    'moving_rate': 0.9
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
