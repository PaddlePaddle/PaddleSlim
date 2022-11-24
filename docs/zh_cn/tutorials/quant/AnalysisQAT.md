# QAT量化分析工具详细教程

## 1. 量化分析工具功能
精度误差分析(metric_error_analyse)：
 - 遍历量化训练后模型的每层，去掉量化节点并计算当前层不量化的模型精度。该功能可以定位具体某层导致的量化损失。



## 2. paddleslim.quant.AnalysisQAT 可传入参数解析
```yaml
quant_model_dir
float_model_dir
model_filename: None
params_filename: None
quantizable_op_type: ["conv2d", "depthwise_conv2d", "mul"]
qat_metric: None
eval_function: None
data_loader: None
save_dir: 'analysis_results'
resume: False
```
- quant_model_dir: 必须传入的量化后的模型文件路径，可为文件夹名
- float_model_dir: 必须传入的量化前的模型文件路径，可为文件夹名
- model_filename: 默认为None，若model_dir为文件夹名，则必须传入以'.pdmodel'结尾的模型名称，若model_dir为'.onnx'模型文件名称，则不需要传入。
- params_filename: 默认为None，若model_dir为文件夹名，则必须传入以'.pdiparams'结尾的模型名称，若model_dir为'.onnx'模型文件名称，则不需要传入。
- quantizable_op_type: 需分析的量化的op类型，默认为`conv2d`, `depthwise_conv2d`, `mul`。
- eval_function：需要传入自定义的验证函数。
- data_loader：模型校准时使用的数据，DataLoader继承自`paddle.io.DataLoader`。可以直接使用模型套件中的DataLoader，或者根据[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)自定义所需要的DataLoader。
- save_dir：分析后保存模型精度或pdf等文件的文件夹，默认为`analysis_results`。
- resume：是否加载中间分析文件




## 3. 量化分析工具的使用
**创建量化分析工具** ：
```
analyzer = AnalysisQAT(
    quant_model_dir=config["quant_model_dir"],
    float_model_dir=config["float_model_dir"],
    model_filename=config["model_filename"],
    params_filename=config["params_filename"],
    quantizable_op_type=config['quantizable_op_type'],
    qat_metric=config['qat_metric'],
    eval_function=eval_function,
    data_loader=eval_loader,
    save_dir=config['save_dir'],
    resume=config['resume'],
)
```


**精度误差分析**
```
analyzer.metric_error_analyse()
```
调用该接口，会遍历量化模型中的每一层，去掉量化节点并计算当前层不量化的模型精度。调用该接口时，需要输入Eval Function。会产出所有去掉一层量化的模型精度排序，将默认保存在 `./analysis_results/analysis.txt` 中。
