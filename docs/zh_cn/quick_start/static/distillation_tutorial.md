#  知识蒸馏

该教程以图像分类模型MobileNetV1为例，说明如何快速使用[PaddleSlim的知识蒸馏接口](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/dist/single_distiller_api.html)。
该示例包含以下步骤：

1. 导入依赖
2. 定义student_program和teacher_program
3. 选择特征图
4. 合并program(merge)并添加蒸馏loss
5. 模型训练

以下章节依次介绍每个步骤的内容。

## 1. 导入依赖

PaddleSlim依赖Paddle2.0版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:

```
import paddle
import numpy as np
import paddleslim as slim
paddle.enable_static()
```

## 2. 定义student_program和teacher_program

本教程在CIFAR数据集上进行知识蒸馏的训练和验证，输入图片尺寸为`[3, 32, 32]`，输出类别数为10。
选择`ResNet50`作为teacher对`MobileNet`结构的student进行蒸馏训练。

```python
model = slim.models.MobileNet()
student_program = paddle.static.Program()
student_startup = paddle.static.Program()
with paddle.static.program_guard(student_program, student_startup):
    image = paddle.static.data(
        name='image', shape=[None, 3, 32, 32], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    gt = paddle.reshape(label, [-1, 1])
    out = model.net(input=image, class_dim=10)
    cost = paddle.nn.functional.loss.cross_entropy(input=out, label=gt)
    avg_cost = paddle.mean(x=cost)
    acc_top1 = paddle.metric.accuracy(input=out, label=gt, k=1)
    acc_top5 = paddle.metric.accuracy(input=out, label=gt, k=5)
```



```python
teacher_model = slim.models.ResNet50()
teacher_program = paddle.static.Program()
teacher_startup = paddle.static.Program()
with paddle.static.program_guard(teacher_program, teacher_startup):
    with paddle.utils.unique_name.guard():
        image = paddle.static.data(
            name='image', shape=[None, 3, 32, 32], dtype='float32')
        predict = teacher_model.net(image, class_dim=10)
exe = paddle.static.Executor(paddle.CPUPlace())
exe.run(teacher_startup)
```

## 3. 选择特征图

我们可以用student_的list_vars方法来观察其中全部的Tensor，从中选出一个或多个变量（Tensor）来拟合teacher相应的变量。

```python
# get all student tensor
student_vars = []
for v in student_program.list_vars():
    student_vars.append((v.name, v.shape))
#uncomment the following lines to observe student's tensor for distillation
#print("="*50+"student_model_vars"+"="*50)
#print(student_vars)

# get all teacher tensor
teacher_vars = []
for v in teacher_program.list_vars():
    teacher_vars.append((v.name, v.shape))
#uncomment the following lines to observe teacher's tensor for distillation
#print("="*50+"teacher_model_vars"+"="*50)
#print(teacher_vars)
```

经过筛选我们可以看到，teacher_program中的'bn5c_branch2b.output.1.tmp_3'和student_program的'depthwise_conv2d_11.tmp_0'尺寸一致，可以组成蒸馏损失函数。

## 4. 合并program (merge)并添加蒸馏loss
merge操作将student_program和teacher_program中的所有Tensor和Op都将被添加到同一个Program中，同时为了避免两个program中有同名变量会引起命名冲突，merge也会为teacher_program中的Tensor添加一个同一的命名前缀name_prefix，其默认值是'teacher_'

为了确保teacher网络和student网络输入的数据是一样的，merge操作也会对两个program的输入数据层进行合并操作，所以需要指定一个数据层名称的映射关系data_name_map，key是teacher的输入数据名称，value是student的

```python
data_name_map = {'image': 'image'}
main = slim.dist.merge(teacher_program, student_program, data_name_map, paddle.CPUPlace())
with paddle.static.program_guard(student_program, student_startup):
    l2_loss = slim.dist.l2_loss('teacher_bn5c_branch2b.output.1.tmp_3', 'depthwise_conv2d_11.tmp_0', student_program)
    loss = l2_loss + avg_cost
    opt = paddle.optimizer.Momentum(0.01, 0.9)
    opt.minimize(loss)
exe.run(student_startup)
```

## 5. 模型训练

为了快速执行该示例，我们选取简单的CIFAR数据，Paddle框架的`paddle.vision.datasets.Cifar10`包定义了CIFAR10数据的下载和读取。 代码如下：

```python
import paddle.vision.transforms as T
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = paddle.vision.datasets.Cifar10(
    mode="train", backend="cv2", transform=transform)

train_loader = paddle.io.DataLoader(
    train_dataset,
    places=paddle.CPUPlace(),
    feed_list=[image, label],
    drop_last=True,
    batch_size=64,
    return_list=False,
    shuffle=True)
```

```python
for idx, data in enumerate(train_loader):
    acc1, acc5, loss_np = exe.run(student_program, feed=data, fetch_list=[acc_top1.name, acc_top5.name, loss.name])
    print("Acc1: {:.6f}, Acc5: {:.6f}, Loss: {:.6f}".format(acc1.mean(), acc5.mean(), loss_np.mean()))
```
