# 外部如何自定义强化学习Controller

首先导入必要的依赖:
```python
### 引入强化学习Controller基类函数和注册类函数
from paddleslim.common.rl_controller.utils import RLCONTROLLER
from paddleslim.common.rl_controller import RLBaseController
```

通过装饰器的方式把自定义强化学习Controller注册到PaddleSlim，继承基类之后需要重写基类中的`next_tokens`和`update`两个函数。注意：本示例仅说明一些必不可少的步骤，并不能直接运行，完整代码请参考[这里]()

```python
### 注意: 类名一定要全部大写
@RLCONTROLLER.register
class LSTM(RLBaseController):
    def __init__(self, range_tables, use_gpu=False, **kwargs):
        ### range_tables 表示tokens的取值范围
        self.range_tables = range_tables
        ### use_gpu 表示是否使用gpu来训练controller
        self.use_gpu = use_gpu
        ### 定义一些强化学习算法中需要的参数
        ...
        ### 构造相应的program, _build_program这个函数会构造两个program，一个是pred_program，一个是learn_program， 并初始化参数
        self._build_program()
        self.place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.exe.run(fluid.default_startup_program())

        ### 保存参数到一个字典中，这个字典由server端统一维护更新，因为可能有多个client同时更新一份参数，所以这一步必不可少，由于pred_program和learn_program使用的同一份参数，所以只需要把learn_program中的参数放入字典中即可
        self.param_dicts = {}
        self.param_dicts.update(self.learn_program: self.get_params(self.learn_program))

    def next_tokens(self, states, params_dict):
        ### 把从server端获取参数字典赋值给当前要用到的program
        self.set_params(self.pred_program, params_dict, self.place)
        ### 根据states构造输入
        self.num_archs = states
        feed_dict = self._create_input()
        ### 获取当前token
        actions = self.exe.run(self.pred_program, feed=feed_dict, fetch_list=self.tokens)
        ...
        return actions

    def update(self, rewards, params_dict=None):
        ### 把从server端获取参数字典赋值给当前要用到的program
        self.set_params(self.learn_program, params_dict, self.place)
        ### 根据`next_tokens`中的states和`update`中的rewards构造输入
        feed_dict = self._create_input(is_test=False, actual_rewards = rewards)
        ### 计算当前step的loss
        loss = self.exe.run(self.learn_program, feed=feed_dict, fetch_list=[self.loss])
        ### 获取当前program的参数并返回，client会把本轮的参数传给server端进行参数更新
        params_dict = self.get_params(self.learn_program)
        return params_dict
```
