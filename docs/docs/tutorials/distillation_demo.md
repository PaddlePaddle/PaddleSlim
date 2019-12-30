本示例将介绍如何使用PaddleSlim蒸馏接口来对模型进行蒸馏训练

## 接口介绍

请参考[蒸馏API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/)。

## PaddleSlim蒸馏训练流程

一般情况下，模型参数量越多，结构越复杂，其性能越好，但运算量和资源消耗也越大。**知识蒸馏** 就是一种将大模型学习到的有用信息（Dark Knowledge）压缩进更小更快的模型，而获得可以匹敌大模型结果的方法。

在本示例中精度较高的大模型被称为teacher，精度稍逊但速度更快的小模型被称为student。

### 1. 定义student_program

```python
student_program = fluid.Program()
student_startup = fluid.Program()
with fluid.program_guard(student_program, student_startup):
    image = fluid.data(
        name='image', shape=[None] + [3, 224, 224], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    # student model definition
    model = MobileNet()
    out = model.net(input=image, class_dim=1000)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
```

### 2. 定义teacher_program

在定义好teacher_program后，可以一并加载训练好的pretrained_model

在teacher_program内需要加上`with fluid.unique_name.guard():`，保证teacher的变量命名不被student_program影响，从而跟能够正确地加载预训练参数

```python
teacher_program = fluid.Program()
teacher_startup = fluid.Program()
with fluid.program_guard(teacher_program, teacher_startup):
    with fluid.unique_name.guard():
        image = fluid.data(
            name='data', shape=[None] + [3, 224, 224], dtype='float32')
        # teacher model definition
        teacher_model = ResNet()
        predict = teacher_model.net(image, class_dim=1000)
exe.run(teacher_startup)
def if_exist(var):
    return os.path.exists(
        os.path.join("./pretrained", var.name)
fluid.io.load_vars(
    exe,
    "./pretrained",
    main_program=teacher_program,
    predicate=if_exist)
```

### 3.选择特征图

定义好student_program和teacher_program后，我们需要从中两两对应地挑选出若干个特征图，留待后续为其添加知识蒸馏损失函数

```python
# get all student variables
student_vars = []
for v in student_program.list_vars():
    try:
        student_vars.append((v.name, v.shape))
    except:
        pass
print("="*50+"student_model_vars"+"="*50)
print(student_vars)
# get all teacher variables
teacher_vars = []
for v in teacher_program.list_vars():
    try:
        teacher_vars.append((v.name, v.shape))
    except:
        pass
print("="*50+"teacher_model_vars"+"="*50)
print(teacher_vars)
```

### 4. 合并Program（merge）

PaddlePaddle使用Program来描述计算图，为了同时计算student和teacher两个Program，这里需要将其两者合并（merge）为一个Program。

merge过程操作较多，具体细节请参考[merge API文档](https://paddlepaddle.github.io/PaddleSlim/api/single_distiller_api/#merge)。

```python
data_name_map = {'data': 'image'}
student_program = merge(teacher_program, student_program, data_name_map, place)
```

### 5.添加蒸馏loss

在添加蒸馏loss的过程中，可能还会引入部分变量（Variable），为了避免命名重复这里可以使用`with fluid.name_scope("distill"):`为新引入的变量加一个命名作用域

```python
with fluid.program_guard(student_program, student_startup):
    with fluid.name_scope("distill"):
        distill_loss = l2_loss('teacher_bn5c_branch2b.output.1.tmp_3', 'depthwise_conv2d_11.tmp_0', main)
        distill_weight = 1
        loss = avg_cost + distill_loss * distill_weight
        opt = create_optimizer()
        opt.minimize(loss)
exe.run(student_startup)
```

至此，我们就得到了用于蒸馏训练的student_program，后面就可以使用一个普通program一样对其开始训练和评估
