#  Knowledge Distillation for Image Classification

In this tutorial, you will learn how to use knowledge distillation API of PaddleSlim
by a demo of MobileNetV1 model on MNIST dataset. This tutorial following workflow:

1. Import dependency
2. Define student_program and teacher_program
3. Select feature maps
4. Merge program and add distillation loss
5. Train distillation model

## 1. Import dependency

PaddleSlim dependents on Paddle1.7. Please ensure that you have installed paddle correctly. Import Paddle and PaddleSlim as below:

```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
```

## 2. Define student_program and teacher_program

This tutorial trains and verifies distillation model on the MNIST dataset. The input image shape is `[1, 28, 28] `and the number of output categories is 10.
Select `ResNet50` as the teacher to perform distillation training on the students of the` MobileNet` architecture.

```python
model = slim.models.MobileNet()
student_program = fluid.Program()
student_startup = fluid.Program()
with fluid.program_guard(student_program, student_startup):
    image = fluid.data(
        name='image', shape=[None] + [1, 28, 28], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    out = model.net(input=image, class_dim=10)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
```



```python
model = slim.models.ResNet50()
teacher_program = fluid.Program()
teacher_startup = fluid.Program()
with fluid.program_guard(teacher_program, teacher_startup):
    with fluid.unique_name.guard():
        image = fluid.data(
            name='image', shape=[None] + [1, 28, 28], dtype='float32')
        predict = teacher_model.net(image, class_dim=10)
exe = fluid.Executor(fluid.CPUPlace())
exe.run(teacher_startup)
```

## 3. Select feature maps

We can use the student_program's list_vars method to observe all the Variables, and select one or more variables from it to fit the corresponding variables of the teacher.

```python
# get all student variables
student_vars = []
for v in student_program.list_vars():
    student_vars.append((v.name, v.shape))
#uncomment the following lines to observe student's variables for distillation
#print("="*50+"student_model_vars"+"="*50)
#print(student_vars)

# get all teacher variables
teacher_vars = []
for v in teacher_program.list_vars():
    teacher_vars.append((v.name, v.shape))
#uncomment the following lines to observe teacher's variables for distillation
#print("="*50+"teacher_model_vars"+"="*50)
#print(teacher_vars)
```

we can see that the shape of 'bn5c_branch2b.output.1.tmp_3' in the teacher_program and the 'depthwise_conv2d_11.tmp_0' of the student are the same and can form the distillation loss function.

## 4. Merge program and add distillation loss
The merge operation adds all Variables and Ops in teacher_program to student_Program. At the same time, in order to avoid naming conflicts caused by variables with the same name in two programs, merge will also add a unified naming prefix **name_prefix** for Variables in teacher_program, which The default value is 'teacher_'_.

In order to ensure that the data of the teacher network and the student network are the same, the merge operation also merges the input data layers of the two programs, so you need to specify a data layer name mapping ***data_name_map***, where key is the input data name of the teacher, and value Is student's.

```python
data_name_map = {'image': 'image'}
main = slim.dist.merge(teacher_program, student_program, data_name_map, fluid.CPUPlace())
with fluid.program_guard(student_program, student_startup):
    l2_loss = slim.dist.l2_loss('teacher_bn5c_branch2b.output.1.tmp_3', 'depthwise_conv2d_11.tmp_0', student_program)
    loss = l2_loss + avg_cost
    opt = fluid.optimizer.Momentum(0.01, 0.9)
    opt.minimize(loss)
exe.run(student_startup)
```

## 5. Train distillation model

The package `paddle.dataset.mnist` of Paddle define the downloading and reading of MNIST dataset.
Define training data reader and test data reader as below：

```python
train_reader = paddle.fluid.io.batch(
    paddle.dataset.mnist.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(['image', 'label'], fluid.CPUPlace(), student_program)
```

Excute following code to run an `epoch` training:


```python
for data in train_reader():
    acc1, acc5, loss_np = exe.run(student_program, feed=train_feeder.feed(data), fetch_list=[acc_top1.name, acc_top5.name, loss.name])
    print("Acc1: {:.6f}, Acc5: {:.6f}, Loss: {:.6f}".format(acc1.mean(), acc5.mean(), loss_np.mean()))
```
