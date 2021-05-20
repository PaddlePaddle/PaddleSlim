单进程蒸馏
=========

merge
---------

.. py:function:: paddleslim.dist.merge(teacher_program, student_program, data_name_map, place, scope=None, name_prefix='teacher_')

`[源代码] <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py#L19>`_

将teacher_program融合到student_program中。

在融合的program中，可以方便地联合原本两个Program中的Tensor做计算。

**参数：**

- **teacher_program** (Program)-定义了teacher模型的 `paddle program <https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program>`_
- **student_program** (Program)-定义了student模型的 `paddle program <https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/Program_cn.html#program>`_
- **data_name_map** (dict)-teacher输入接口名与student输入接口名的映射，其中dict的 *key* 为teacher的输入名，*value* 为student的输入名
- **place** (fluid.CPUPlace()|fluid.CUDAPlace(N))-该参数表示程序运行在何种设备上，这里的N为GPU对应的ID
- **scope** (Scope)-该参数表示程序使用的变量作用域，如果不指定将使用默认的全局作用域 `global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/paddle_cn/global_scope_cn.html#global-scope>`_ 。默认值： None
- **name_prefix** (str)-为避免同名参数冲突，merge操作将统一为teacher的 `Variables <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/beginners_guide/basic_concept/variable.html#variable>`_ 添加的名称前缀name_prefix。默认值：teacher_

**返回：** 无

**使用示例：**

.. code-block:: python

   import paddle
   import paddle.fluid as fluid
   import paddleslim.dist as dist
   paddle.enable_static()
   student_program = fluid.Program()
   with fluid.program_guard(student_program):
       x = fluid.layers.data(name='x', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(x, 32, 1)
       out = fluid.layers.conv2d(conv, 64, 3, padding=1)
   teacher_program = fluid.Program()
   with fluid.program_guard(teacher_program):
       y = fluid.layers.data(name='y', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(y, 32, 1)
       conv = fluid.layers.conv2d(conv, 32, 3, padding=1)
       out = fluid.layers.conv2d(conv, 64, 3, padding=1)
   data_name_map = {'y':'x'}
   USE_GPU = False
   place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()
   dist.merge(teacher_program, student_program,
                             data_name_map, place)


fsp_loss
---------

.. py:function:: paddleslim.dist.fsp_loss(teacher_var1_name, teacher_var2_name, student_var1_name, student_var2_name, program=None)

`[源代码] <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py#L90>`_

为program内的teacher var和student var添加fsp_loss.

fsp_loss出自论文 `A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning <http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf>`_

**参数：**

- **teacher_var1_name** (str): teacher_var1的名称. 对应的variable是一个形为 ``[batch_size, x_channel, height, width]`` 的4-D特征图Tensor，数据类型为float32或float64
- **teacher_var2_name** (str): teacher_var2的名称. 对应的variable是一个形为 ``[batch_size, y_channel, height, width]`` 的4-D特征图Tensor，数据类型为float32或float64。只有y_channel可以与teacher_var1的x_channel不同，其他维度必须与teacher_var1相同
- **student_var1_name** (str): student_var1的名称. 对应的variable需与teacher_var1尺寸保持一致，是一个形为 ``[batch_size, x_channel, height, width]`` 的4-D特征图Tensor，数据类型为float32或float64
- **student_var2_name** (str): student_var2的名称. 对应的variable需与teacher_var2尺寸保持一致，是一个形为 ``[batch_size, y_channel, height, width]`` 的4-D特征图Tensor，数据类型为float32或float64。只有y_channel可以与student_var1的x_channel不同，其他维度必须与student_var1相同
- **program** (Program): 用于蒸馏训练的fluid program, 如果未指定则使用 `fluid.default_main_program() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/fluid_cn/default_main_program_cn.html#default-main-program>`_ 。默认值：None

**返回：**

- (Variable): 由teacher_var1, teacher_var2, student_var1, student_var2组合得到的fsp_loss

**使用示例：**

.. code-block:: python

   import paddle
   import paddle.fluid as fluid
   import paddleslim.dist as dist
   paddle.enable_static()
   student_program = fluid.Program()
   with fluid.program_guard(student_program):
       x = fluid.layers.data(name='x', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(x, 32, 1, name='s1')
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='s2')
   teacher_program = fluid.Program()
   with fluid.program_guard(teacher_program):
       y = fluid.layers.data(name='y', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(y, 32, 1, name='t1')
       conv = fluid.layers.conv2d(conv, 32, 3, padding=1)
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='t2')
   data_name_map = {'y':'x'}
   USE_GPU = False
   place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()
   dist.merge(teacher_program, student_program, data_name_map, place)
   with fluid.program_guard(student_program):
       distillation_loss = dist.fsp_loss('teacher_t1.tmp_1', 'teacher_t2.tmp_1',
                                         's1.tmp_1', 's2.tmp_1', student_program)



l2_loss
------------

.. py:function:: paddleslim.dist.l2_loss(teacher_var_name, student_var_name, program=None)

`[源代码] <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py#L118>`_

为program内的teacher var和student var添加l2 loss

**参数：**

- **teacher_var_name** (str): teacher_var的名称.
- **student_var_name** (str): student_var的名称.
- **program** (Program): 用于蒸馏训练的fluid program。如果未指定则使用 `fluid.default_main_program() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/fluid_cn/default_main_program_cn.html#default-main-program>`_ 。默认值：None

**返回：**

- (Variable): 由teacher_var, student_var组合得到的l2_loss

**使用示例：**

.. code-block:: python

   import paddle 
   import paddle.fluid as fluid
   import paddleslim.dist as dist
   paddle.enable_static()
   student_program = fluid.Program()
   with fluid.program_guard(student_program):
       x = fluid.layers.data(name='x', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(x, 32, 1, name='s1')
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='s2')
   teacher_program = fluid.Program()
   with fluid.program_guard(teacher_program):
       y = fluid.layers.data(name='y', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(y, 32, 1, name='t1')
       conv = fluid.layers.conv2d(conv, 32, 3, padding=1)
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='t2')
   data_name_map = {'y':'x'}
   USE_GPU = False
   place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()
   dist.merge(teacher_program, student_program, data_name_map, place)
   with fluid.program_guard(student_program):
       distillation_loss = dist.l2_loss('teacher_t2.tmp_1', 's2.tmp_1',
                                        student_program)



soft_label_loss
-------------------

.. py:function:: paddleslim.dist.soft_label_loss(teacher_var_name, student_var_name, program=None, teacher_temperature=1., student_temperature=1.)

`[源代码] <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py#L136>`_

为program内的teacher var和student var添加soft label loss

soft_label_loss出自论文 `Distilling the Knowledge in a Neural Network <https://arxiv.org/pdf/1503.02531.pdf>`_

**参数：**

- **teacher_var_name** (str): teacher_var的名称.
- **student_var_name** (str): student_var的名称.
- **program** (Program): 用于蒸馏训练的fluid program。如果未指定则使用 `fluid.default_main_program() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/fluid_cn/default_main_program_cn.html#default-main-program>`_ 。默认值：None
- **teacher_temperature** (float): 对teacher_var进行soft操作的温度值，温度值越大得到的特征图越平滑
- **student_temperature** (float): 对student_var进行soft操作的温度值，温度值越大得到的特征图越平滑

**返回：**

- (Variable): 由teacher_var, student_var组合得到的soft_label_loss

**使用示例：**

.. code-block:: python

   import paddle
   import paddle.fluid as fluid
   import paddleslim.dist as dist
   paddle.enable_static()
   student_program = fluid.Program()
   with fluid.program_guard(student_program):
       x = fluid.layers.data(name='x', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(x, 32, 1, name='s1')
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='s2')
   teacher_program = fluid.Program()
   with fluid.program_guard(teacher_program):
       y = fluid.layers.data(name='y', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(y, 32, 1, name='t1')
       conv = fluid.layers.conv2d(conv, 32, 3, padding=1)
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='t2')
   data_name_map = {'y':'x'}
   USE_GPU = False
   place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()
   dist.merge(teacher_program, student_program, data_name_map, place)
   with fluid.program_guard(student_program):
       distillation_loss = dist.soft_label_loss('teacher_t2.tmp_1',
                                                's2.tmp_1', student_program, 1., 1.)



loss
--------

.. py:function:: paddleslim.dist.loss(loss_func, program=None, **kwargs)

`[源代码] <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/dist/single_distiller.py#L165>`_

支持对teacher_var和student_var使用任意自定义损失函数

**参数：**

- **loss_func** (python function): 自定义的损失函数，输入为teacher var和student var，输出为自定义的loss
- **program** (Program): 用于蒸馏训练的fluid program。如果未指定则使用 `fluid.default_main_program() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/fluid_cn/default_main_program_cn.html#default-main-program>`_ 。默认值：None
- **kwargs** : loss_func输入名与对应variable名称

**返回：**

- (Variable): 自定义的损失函数loss

**使用示例：**

.. code-block:: python

   import paddle
   import paddle.fluid as fluid
   import paddleslim.dist as dist
   paddle.enable_static()
   student_program = fluid.Program()
   with fluid.program_guard(student_program):
       x = fluid.layers.data(name='x', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(x, 32, 1, name='s1')
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='s2')
   teacher_program = fluid.Program()
   with fluid.program_guard(teacher_program):
       y = fluid.layers.data(name='y', shape=[1, 28, 28])
       conv = fluid.layers.conv2d(y, 32, 1, name='t1')
       conv = fluid.layers.conv2d(conv, 32, 3, padding=1)
       out = fluid.layers.conv2d(conv, 64, 3, padding=1, name='t2')
   data_name_map = {'y':'x'}
   USE_GPU = False
   place = fluid.CUDAPlace(0) if USE_GPU else fluid.CPUPlace()
   dist.merge(teacher_program, student_program, data_name_map, place)
   def adaptation_loss(t_var, s_var):
       teacher_channel = t_var.shape[1]
       s_hint = fluid.layers.conv2d(s_var, teacher_channel, 1)
       hint_loss = fluid.layers.reduce_mean(fluid.layers.square(s_hint - t_var))
       return hint_loss
   with fluid.program_guard(student_program):
       distillation_loss = dist.loss(adaptation_loss, student_program,
               t_var='teacher_t2.tmp_1', s_var='s2.tmp_1')

.. note::

    在添加蒸馏loss时会引入新的variable，需要注意新引入的variable不要与student variables命名冲突。这里建议两种用法（两种方法任选其一即可）：

    1. 建议与student_program使用同一个命名空间，以避免一些未指定名称的variables(例如tmp_0, tmp_1...)多次定义为同一名称出现命名冲突

    2. 建议在添加蒸馏loss时指定一个命名空间前缀，具体用法请参考Paddle官方文档 `fluid.name_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/name_scope_cn.html#name-scope>`_
