# RV1109在线量化步骤
##静态图模型离线融合conv层和bn层

###准备float32预测模型
    使用save_inference_model保存得到。
    
###融合float32模型的权重

    使用 `fuse_conv_bn.py` 加载fp32预测模型，进行fuse，得到融合conv+bn后的预测模型权重，该权重可以作为预训练的权重进行量化训练。

    `python fuse_conv_bn.py --load_model_dir ** --load_model_filename ** --load_params_filename ** --save_model_dir **`

###修改代码融合conv层和bn层

    为了可以加载已经融合conv层和bn层的预训练权重，需要修改组网代码。
    对conv层包含bias和不包含bias两种情况，分别进行处理。
    
1. 当conv2d带有bias，如下例子所示：
* 将bn删掉，将conv2d的输出连到下一个op；
* 如果bn带有激活，需要将激活加到conv2d中。

```
# 修改前
conv1 = fluid.layers.conv2d(input=data, num_filters=32, filter_size=3,
		stride=1, padding=1)  # bias_attr is None in default
bn1 = fluid.layers.batch_norm(input=conv1, act='relu')
fc1 = fluid.layers.fc(input=bn1, size=10)
```

```
# 修改后
conv1 = fluid.layers.conv2d(input=data, num_filters=32, filter_size=3,
		stride=1, padding=1, act='relu')  # bias_attr is None in default
fc1 = fluid.layers.fc(input=conv1, size=10)
```

2. 当conv2d不带有bias，如下例子所示：
* conv+bn融合为conv+elementwise_add，conv不用修改，删除bn，添加elementwise_add
* elementwise_add的输入权重必须按照如下规则设置name:
    * eleadd_y_name = "fuse_conv_bn/conv2d_eltwise_y_in/" + str(index)
    * index是整个网络中依次出现conv_bn的序号(重要)
* elementwise_add输入权重的shape设置为conv2d或者depthwise_conv2d的输出通道数
* elementwise_add的axis设置为1
* 如果bn有激活函数，需要在elementwise_add中添加对应激活函数

注意，上述conv2d不带有bias的修改方法，不适用于depthwise_conv2d + bn的融合。

```
# 修改前
conv = fluid.layers.conv2d(
	input=input,
	num_filters=32,
	filter_size=3,
	stride=1,
	padding=1,
	groups=num_groups,
	act=None,
	bias_attr=False)
bn = fluid.layers.batch_norm(
	input=conv,
	act='relu',
	param_attr=ParamAttr(name=name + "_bn" + "_scale"),
	bias_attr=ParamAttr(name=name + "_bn" + "_offset"),
	moving_mean_name=name + "_bn" + '_mean',
	moving_variance_name=name + "_bn" + '_variance')
fc = fluid.layers.fc(input=bn, size=10)	
```

```
# 修改后

conv = fluid.layers.conv2d(
	input=input,
	num_filters=32,
	filter_size=3,
	stride=1,
	padding=1,
	groups=num_groups,
	act=None,
	bias_attr=False)
eleadd_y_name = "fuse_conv_bn/conv2d_eltwise_y_in/" + str(conv2d_idx）
conv2d_idx += 1
eleadd_y = fluid.layers.create_parameter(
			name=eleadd_y_name,
			shape=[32],
			dtype='float32')
eleadd = fluid.layers.elementwise_add(
		x=conv,
		y=eleadd_y,
		axis=1,
		act='relu')
fc = fluid.layers.fc(input=eleadd, size=10)	
```


### 测试fuse和修改代码是否正确

	基于修改的组网代码，构建训练program和测试program，加载融合conv+bn后的预测模型权重，按照常规方法对测试program进行测试，验证精度是否合理。

###量化训练

    基于修改的组网代码，构建训练program和测试program，加载融合conv+bn后的预测模型权重，按照常规方法调用paddleslim中量化训练api，进行量化训练，产出量化模型。

    因为融合了conv+bn，所以学习率需要适当调小，比如1e-5。

    训练多个epoch，选用精度最高的量化模型。
