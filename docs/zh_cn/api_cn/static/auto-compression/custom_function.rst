如何基于Paddle自定义DataLoader
==========

可以参考飞桨官网: 
    1. `自定义数据集 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/02_data_load_cn.html#erzidingyishujuji>`_
    2. `数据加载 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/02_data_load_cn.html#sanshujujiazai>`_



如何基于Paddle自定义测试回调函数
==========

1. 输入输出格式
-----------------

自定义测试回调函数的输入和输出是固定的。

1.1 输入
##########

回调函数必须有以下4个输入：

**executor**: 飞桨的执行器，执行器可以用来执行指定的 ``Program`` 或者 ``CompiledProgram`` 。
**program**: 飞桨对计算图的一种静态描述。
**feed_name_list**: 所需提供数据的所有变量名称（即所有输入变量的名称）。
**fetch_targets**: 包含模型的所有输出变量。通过这些输出变量即可得到模型的预测结果。

1.2 输出
##########

回调函数必须有1个输入：

**result(float)**: 模型的计算指标，仅返回最重要的指标即可，返回的指标用来判断是否数据读取是否正确，和训练过程中是否达到了设定的优化目标。

1.3 自定义计算逻辑
##########

首先需要根据 `如何基于Paddle自定义DataLoader <>`_ 章节定义测试数据集 ``test_dataloader`` 。

```python

### 定义包含几个固定输入的测试函数。
def eval_function(exe, program, feed_name_list, fetch_targets):
    results = []
    ### 遍历数据集
    for data in test_dataloader():
        ### 从数据集中提取出label
        labels = data.pop('label')
        ### 传入实际数据，运行计算图，得到输出
        outputs = exe.run(program, feed=data, fetch_list=fetch_targets)
        ### 根据输出结果和label信息计算当前批次数据指标
        result.append(metric(outputs, labels))
    ### 返回float类型的整体指标
    return np.mean(results)

```
