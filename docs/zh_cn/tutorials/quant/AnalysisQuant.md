# 量化分析工具详细教程

## 1. 量化分析工具功能
1. 统计分析(statistical_analyse)：
    - 可视化激活和权重箱状图。箱状图可发现是否出现离群点。
    - 可视化权重和激活直方分布图。直方分布图可观察更具体的数值分布。
    - 提供量化前后权重和激活的具体数据信息，包括min，max，mean，std等

2. 精度误差分析(metric_error_analyse)：
    - 遍历量化模型的每层，并计算量化后精度。该功能可以定位具体某层导致的量化损失。

3. 获取目标模型(get_target_quant_model)：
    - 输入预期精度，直接产出符合预期精度的量化模型。


## 2. paddleslim.quant.AnalysisQuant 可传入参数解析
```yaml
model_dir
model_filename: None
params_filename: None
eval_function: None
data_loader: None
save_dir: 'analysis_results'
resume: False
ptq_config
```
- model_dir: 必须传入的模型文件路径，可为文件夹名；若模型为ONNX类型，直接输入'.onnx'模型文件名称即可。
- model_filename: 默认为None，若model_dir为文件夹名，则必须传入以'.pdmodel'结尾的模型名称，若model_dir为'.onnx'模型文件名称，则不需要传入。
- params_filename: 默认为None，若model_dir为文件夹名，则必须传入以'.pdiparams'结尾的模型名称，若model_dir为'.onnx'模型文件名称，则不需要传入。
- eval_function：若需要验证精度，需要传入自定义的验证函数。
- data_loader：模型校准时使用的数据，DataLoader继承自`paddle.io.DataLoader`。可以直接使用模型套件中的DataLoader，或者根据[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)自定义所需要的DataLoader。
- save_dir：分析后保存模型精度或pdf等文件的文件夹，默认为`analysis_results`。
- resume：是否加载中间分析文件
- ptq_config：可传入的离线量化中的参数，详细可参考[离线量化文档](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)。




## 3. 量化分析工具的使用
**创建量化分析工具** ：
```
analyzer = AnalysisQuant(
		model_dir=config["model_dir"],
		model_filename=config["model_filename"],
		params_filename=config["params_filename"],
		eval_function=eval_function,
		data_loader=data_loader,
		save_dir=config['save_dir'],
		ptq_config=config['PTQ'])
```

**统计分析**
```
analyzer.statistical_analyse()
```

调用该接口，会统计量化前和量化后每一个可量化权重和其对应激活的数据。只使用该接口可以不输入Eval Function，但需要输入DataLoader，少量数据即可。会产出以下文件：
- `fp_activation_boxplot.pdf`：量化前Float数据类型的模型激活箱状图
- `fp_weight_boxplot.pdf`：量化前Float数据类型的模型权重箱状图
- `quantized_activation_boxplot.pdf`：量化后INT数据类型的模型激活箱状图
- `quantized_weight_boxplot.pdf`：量化后INT数据类型的模型权重箱状图
- `fp_activation_histplot.pdf`：量化前Float数据类型的模型激活直方图
- `fp_weight_histplot.pdf`：量化前Float数据类型的模型权重直方图
- `quantized_activation_histplot.pdf`：量化后INT数据类型的模型激活直方图
- `quantized_weight_histplot.pdf`：量化后INT数据类型的模型权重直方图
- `statistic.csv`：量化前后权重和激活的具体数据信息，表格中会保存的信息有：
	- Var Name: Variable的名称
	- Var Type：Variable的类型，Weight或Activation
	- Corresponding Weight Name：如果为Activation，其对应的Weight名称
	- FP32 Min：量化前Float数据类型的最小值
	- FP32 Max：量化前Float数据类型的最大值
	- FP32 Mean：量化前Float数据类型的平均值
	- FP32 Std：量化前Float数据类型的方差值
	- Quantized Min：量化后INT数据类型的最小值
	- Quantized Max：量化后INT数据类型的最大值
	- Quantized Mean：量化后INT数据类型的平均值
	- Quantized Std：量化后INT数据类型的方差值
	- Diff Min：量化前后该Variable的相差的最小值
	- Diff Max：量化前后该Variable的相差的最大值
	- Diff Mean：量化前后该Variable的相差的平均值
	- Diff Std：量化前后该Variable的相差的方差值


**精度误差分析**
```
analyzer.metric_error_analyse()
```
调用该接口，会遍历量化模型中的一层，并计算量化该层后模型的损失。调用该接口时，需要输入Eval Function。会产出所有只量化一层的模型精度排序，将默认保存在 `./analysis_results/analysis.txt` 中。



**直接产出符合预期精度的目标量化模型**
```
analyzer.get_target_quant_model(target_metric)
```

## 4. 根据分析结果执行离线量化
执行完量化分析工具后，可根据 `analysis.txt` 中的精度排序，在量化中去掉效果较差的层，具体操作为：在调用 `paddleslim.quant.quant_post_static` 时加入参数 `skip_tensor_list`，将需要去掉的层传入即可。
