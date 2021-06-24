## 1.生成op单测耗时可执行文件
>编译环境准备：建议使用docker开发环境（在开发机上编译会更快）

进入PaddleLite，执行脚本./lite/tools/build.sh，选择test编译模式，代码举例如下所示：

>./lite/tools/build.sh \
>  --arm_os=android \
>  --arm_abi=armv8 \
>  --build_extra=OFF \
>  --arm_lang=gcc \
>  --android_stl=c++_static \
>  --build_extra=OFF \
>  test
具体arm_os等编译参数含义，见[链接](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/source_compile/#%E7%BC%96%E8%AF%91%E6%A8%A1%E5%BC%8F%E4%B8%8E%E5%8F%82%E6%95%B0)。

编译后，在PaddleLite的根目录下会生成形如 build.lite.${arm_os}.${arm_abi}.${arm_lang}  的文件夹，其包含编译结果。test编译结果在build*/lite/tests目录下，op单测可执行文件（用于测试单个op的耗时，而不是测试计算精度）在build*/lite/tests/benchmark下。

然后将这些可执行文件拷贝到设备端即可执行（前提是设备端链接正常，并且pc机安装了adb）

Mac通过homebrew来安装adb：
>$ brew install android-platform-tools

拷贝命令如下：
>$ adb push <来源目录> <目标目录> #来源目录即为你的pc机上面的目录，目标目录即为设备端上面的目录

如果不知道push到哪里，可以先执行 adb shell 进入设备端查看，退出指令为exit。安卓手机USB连上电脑，打开设置 -> 开启开发者模式 -> 开启USB调试 -> 允许（授权）当前电脑调试手机。

## 2.建立op延时的查找表
benchmark下包含了activation、conv、pooing、bn、fc五类op的可执行单测文件。使用这些可执行单测文件，设置输入的可选参数，在安卓端即可测试相应op的耗时。

首先建立op表，op表如下所示：

>op_name    input_dims     param_info  
>
>conv       [N C H W]    (ch_out=48, stride=1, group=1, kernel=1x1, pad=0, dilation=1, flag_bias=0, flag_act=0, dtype=float)  
>
>activation [N C H W]    (act_type=relu, dtype=float)
>
>batchnorm  [N C H W]    (epsilon=1e-4f, momentum=0.9f, dtype=float)
>
>pooling    [N C H W]    (stride=2, pad=0, kernel=2x2, ceil_mode=0, flag_global=0, exclusive=0, pooling_type=avg, dtype=float)
>
>fc         [N C H W]    (param_dim=64x1000, flag_bias=1, dtype=float)


其中op_name、input_dims和param_info等字段之间以tab隔开，param内部用逗号隔开，in_dims内部用空格隔开。dtype描述该层op使用的数据类型，支持的合法输入为float/int8_float/int8_int8, 现在conv支持三种数据类型，其他op只支持float一种数据类型。

conv中flag_act表示是否融合激活函数，取值为可0/1/2/4，0-none, 1-relu, 2-relu6, 4-leaky relu

activation中act_type表示激活函数类型，合法取值为relu/relu6/leaky_relu/tanh/swish/exp/abs/hard_swish/reciprocal/threshold_relu。

其他参数的详细介绍可参考[链接](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/tests/benchmark)。

PaddleLite/lite/tests/benchmark目录下，有get_latency_lookup_table.py会逐行读取上述查找表，读取op及其参用于执行相应单测文件，测试平均耗时，并将其写入新文件中，最终生成完整的op耗时查找表，形式如下：

>op_name    input_dims   output_dims   param_info  min_latency(ms)   max_latency(ms)   avg_latency(ms)
>
>conv       [N C H W]    [N C H W]     (ch_out=48, stride=1, group=1, kernel=1x1, pad=0, dilation=1, flag_bias=0, flag_act=0, dtype=float)  
>
>activation [N C H W]    [N C H W]     (act_type=relu, dtype=float)
>
>batchnorm  [N C H W]    [N C H W]     (epsilon=1e-4f, momentum=0.9f, dtype=float)
>
>pooling    [N C H W]    [N C H W]     (stride=2, pad=0, kernel=2x2, ceil_mode=0, flag_global=0, exclusive=0, pooling_type=avg, dtype=float)
>
>fc         [N C H W]    [N C H W]     (param_dim=64x1000, flag_bias=1, dtype=float)


## 3.模型耗时评估
get_net_latency.py用于读取查找表文件，累加计算该模型对应的全部op耗时即可得到整个模型的推理耗时。实验测试比较过通过op单测计算而得的模型耗时与直接测试模型耗时相差无几，在模型耗时的波动范围内，详见op单测工具。

对于整个模型的实际耗时测试，编译时需开启[profiling工具](https://paddle-lite.readthedocs.io/zh/develop/user_guides/debug.html)，并参考[链接](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/benchmark_tools/)进行编译。


## 拟实现效果
对某类op进行加速优化后，或是添加修改了相应op，需要对整个模型测试耗时，评估op优化收益，这时可以只对优化后的op进行单测，将其结果替换查重表中相应的op耗时，基于新的查找表累加计算整个模型的耗时，从而无需部署新模型即可得到模型耗时的结果。查找表还可用于NAS。
