# 量化分析工具详细教程

## 1. 量化分析工具功能
1. 遍历模型所有层，依次量化该层，计算量化后精度。为所有只量化一层的模型精度排序，可视化不适合量化的层，以供量化时可选择性跳过不适合量化的层。
2. 可视化量化效果最好和最差的层的权重和激活分布图，以供分析模型量化效果的原因。
3. 【敬请期待】输入预期精度，直接产出符合预期精度的量化模型。

## 2. paddleslim.quant.AnalysisQuant 可传入参数解析
```yaml
model_dir
model_filename: None
params_filename: None
eval_function: None
data_loader: None
save_dir: 'analysis_results'
checkpoint_name: 'analysis_checkpoint.pkl'
num_histogram_plots: 10
ptq_config
```
- model_dir: 必须传入的模型文件路径，可为文件夹名；若模型为ONNX类型，直接输入'.onnx'模型文件名称即可。
- model_filename: 默认为None，若model_dir为文件夹名，则必须传入以'.pdmodel'结尾的模型名称，若model_dir为'.onnx'模型文件名称，则不需要传入。
- params_filename: 默认为None，若model_dir为文件夹名，则必须传入以'.pdiparams'结尾的模型名称，若model_dir为'.onnx'模型文件名称，则不需要传入。
- eval_function：目前不支持为None，需要传入自定义的验证函数。
- data_loader：模型校准时使用的数据，DataLoader继承自`paddle.io.DataLoader`。可以直接使用模型套件中的DataLoader，或者根据[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)自定义所需要的DataLoader。
- save_dir：分析后保存模型精度或pdf等文件的文件夹，默认为`analysis_results`。
- checkpoint_name：由于模型可能存在大量层需要分析，因此分析过程中会中间保存结果，如果程序中断会自动加载已经分析好的结果，默认为`analysis_checkpoint.pkl`。
- num_histogram_plots：需要可视化的直方分布图数量。可视化量化效果最好和最坏的该数量个权重和激活的分布图。默认为10。若不需要可视化直方图，设置为0即可。
- ptq_config：可传入的离线量化中的参数，详细可参考[离线量化文档](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)。




##  3. 量化分析工具产出内容

量化分析工具会默认会产出以下目录：
```
analysis_results/
├── analysis.txt
├── best_weight_hist_result.pdf
├── best_act_hist_result.pdf
├── worst_weight_hist_result.pdf
├── worst_act_hist_result.pdf
```
- 所有只量化一层的模型精度排序，将默认保存在 `./analysis_results/analysis.txt` 中。
- 通过设置参数`num_histogram_plots`，可选择绘出该数量个量化效果最好和最差层的weight和activation的直方分布图，将以PDF形式保存在 `./analysis_results` 文件夹下， 分别保存为 `best_weight_hist_result.pdf`，`best_act_hist_result.pdf`，`worst_weight_hist_result.pdf` 和 `worst_act_hist_result.pdf` 中以供对比分析。


##  3. 根据分析结果执行离线量化
执行完量化分析工具后，可根据 `analysis.txt` 中的精度排序，在量化中去掉效果较差的层，具体操作为：在调用 `paddleslim.quant.quant_post_static` 时加入参数 `skip_tensor_list`，将需要去掉的层传入即可。
