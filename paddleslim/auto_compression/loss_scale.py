#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.common_ops_import import check_type, check_variable_and_dtype, LayerHelper
from paddle.utils import unique_name


def check_finite_and_unscale(x, scale, name=None, float_status=None):
    """
    Check if input X contains all finite data, if yes, scale it by input Scale.
    $$Out = X / scale$$
    If any tensor in X contains Inf or Nan, the Out will generate a indicator.
    FoundInfinite will be 1 (True), and Out will not be scaled. In this case, the data of
    Out should not be used, and its data may not be deterministic.
    Otherwise, FoundInfinite will be 0 (False).
    Args:
        x(list|tuple): The input tensors of check_finite_and_unscale operator.
        scale: The scale of check_finite_and_unscale operator.
        float_status(Tensor): (Only used on NPU) The float status to check overflow.
    """
    check_type(x, 'x', (tuple, list), 'check_finite_and_unscale')
    for e in x:
        check_variable_and_dtype(
            e,
            "x",
            ['float16', 'float32', 'float64'],
            'check_finite_and_unscale', )

    helper = LayerHelper("check_finite_and_unscale", **locals())

    found_inf = helper.create_variable_for_type_inference(dtype='bool')

    inputs = {'X': x, 'Scale': scale}
    outputs = {'Out': x, 'FoundInfinite': found_inf}
    helper.append_op(
        type='check_finite_and_unscale', inputs=inputs, outputs=outputs)

    return x, found_inf


def update_loss_scaling(
        x,
        found_inf,
        prev_loss_scaling,
        num_good_steps,
        num_bad_steps,
        incr_every_n_steps,
        decr_every_n_nan_or_inf,
        incr_ratio,
        decr_ratio,
        stop_update=False,
        name=None, ):
    """
    Update loss scaling according to overall gradients. If all gradients is
    finite after incr_every_n_steps, loss scaling will increase by incr_ratio.
    Otherwise, loss scaling will decrease by decr_ratio after
    decr_every_n_nan_or_inf steps and each step some gradients are infinite.
    Args:
        x(list|tuple): The input tensors of update_loss_scaling operator.
        found_inf (Variable): A boolean variable indicates whether
                                     there is any infinite gradient.
        prev_loss_scaling (Variable): Previous loss scaling.
        num_good_steps (Variable): A variable accumulates good steps in which
                                   all gradients are finite.
        num_bad_steps (Variable): A variable accumulates bad steps in which
                                  some gradients are infinite.
        incr_every_n_steps (int): A variable represents increasing loss
                                       scaling every n consecutive steps with
                                       finite gradients.
        decr_every_n_nan_or_inf (int): A variable represents decreasing
                                            loss scaling every n accumulated
                                            steps with nan or inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing
                           loss scaling.
    """

    check_variable_and_dtype(
        prev_loss_scaling,
        "prev_loss_scaling",
        ['float32', 'float64'],
        "update_loss_scaling", )
    check_type(x, 'x', (tuple, list), 'update_loss_scaling')
    for e in x:
        check_variable_and_dtype(e, "x", ['float16', 'float32', 'float64'],
                                 'update_loss_scaling')
        if e.dtype == paddle.framework.core.VarDesc.VarType.FP16:
            assert (
                prev_loss_scaling.dtype ==
                paddle.framework.core.VarDesc.VarType.FP32
            ), "The dtype of prev_loss_scaling should be float32 when the dtype of x is float16."
        else:
            assert (
                prev_loss_scaling.dtype == e.dtype
            ), "The dtype of prev_loss_scaling should be equal to the dtype of x."

    helper = LayerHelper("update_loss_scaling", **locals())

    inputs = {
        'X': x,
        'FoundInfinite': found_inf,
        'PrevLossScaling': prev_loss_scaling,
        'InGoodSteps': num_good_steps,
        'InBadSteps': num_bad_steps,
    }

    outputs = {
        'Out': x,
        'LossScaling': prev_loss_scaling,
        'OutGoodSteps': num_good_steps,
        'OutBadSteps': num_bad_steps,
    }

    attrs = {
        'incr_every_n_steps': incr_every_n_steps,
        'decr_every_n_nan_or_inf': decr_every_n_nan_or_inf,
        'incr_ratio': incr_ratio,
        'decr_ratio': decr_ratio,
    }

    if isinstance(stop_update, paddle.static.Variable):
        inputs['StopUpdate'] = stop_update
    else:
        attrs['stop_update'] = stop_update

    helper.append_op(
        type='update_loss_scaling', inputs=inputs, outputs=outputs, attrs=attrs)

    return x


class LossScaling:
    def __init__(self,
                 optimizer,
                 init_loss_scaling=2**31,
                 incr_every_n_steps=2,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.5):
        self._optimizer = optimizer
        self._init_loss_scaling = init_loss_scaling
        self._incr_every_n_steps = incr_every_n_steps
        self._decr_every_n_nan_or_inf = decr_every_n_nan_or_inf
        self._incr_ratio = incr_ratio
        self._decr_ratio = decr_ratio
        self._loss_scaling = paddle.static.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=self._init_loss_scaling,
            dtype='float32',
            persistable=True, )
        self._num_good_steps = paddle.static.create_global_var(
            name=unique_name.generate("num_good_steps"),
            shape=[1],
            value=0,
            dtype='int32',
            persistable=True, )
        self._num_bad_steps = paddle.static.create_global_var(
            name=unique_name.generate("num_bad_steps"),
            shape=[1],
            value=0,
            dtype='int32',
            persistable=True, )
        self._train_program = None

    def _supports_check_nan_inf(self):
        return getattr(self._optimizer, "_supports_check_nan_inf", False)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        The first part of ``minimize``, do auto-diff to append backward operations for
        the current program.
        Args:
            loss (Tensor): ``loss`` tensor to run optimizations.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameters (list, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.
            callbacks (list, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.
        Return:
            list: list of (param, grad) tensor pairs, param is ``Parameter``,
                grad is the gradient value corresponding to the parameter.
        """
        program = loss.block.program
        self._train_program = program
        with paddle.static.program_guard(self._train_program, startup_program):
            if loss.dtype != paddle.framework.core.VarDesc.VarType.FP32:
                loss = loss.astype('float32')
            self._scaled_loss = loss * self._loss_scaling
            params_grads = self._optimizer.backward(
                self._scaled_loss,
                startup_program,
                parameter_list,
                no_grad_set,
                callbacks, )
        return params_grads

    def _apply_optimize(self, loss, startup_program, params_grads):
        program = loss.block.program
        with paddle.static.program_guard(program, startup_program):
            optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        opt_dict = self._optimizer.__class__.__dict__
        if 'minimize' in opt_dict and isinstance(opt_dict['minimize'],
                                                 types.FunctionType):
            warnings.warn(
                "The decorated optimizer has its own `minimize` method, but it will not be executed."
            )
        assert isinstance(
            loss, paddle.static.Variable), "The loss should be an Tensor."

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set, )

        optimize_ops = self._apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads

    def _split_grads(self, params_grads):
        grads = [g for _, g in params_grads]
        return grads

    def _check_finite_and_unscale(self, params_grads):
        grads = self._split_grads(params_grads)
        found_infs = []
        with self._train_program._optimized_guard(grads):
            _, found_inf = check_finite_and_unscale(
                grads,
                self._loss_scaling,
                name="find_infinite_scale",
                float_status=None, )
        return found_inf

    def _add_dynamic_loss_scaling(self, params_grads, found_inf):
        if self._supports_check_nan_inf():
            with self._train_program._optimized_guard([]):
                update_loss_scaling(
                    [],
                    found_inf,
                    self._loss_scaling,
                    self._num_good_steps,
                    self._num_bad_steps,
                    self._incr_every_n_steps,
                    self._decr_every_n_nan_or_inf,
                    self._incr_ratio,
                    self._decr_ratio,
                    stop_update=self._optimizer._get_stop_update_var(),
                    name="update_loss_scaling", )
            return

        grads = self._split_grads(params_grads)
        with self._train_program._optimized_guard([]):
            update_loss_scaling(
                grads,
                found_inf,
                self._loss_scaling,
                self._num_good_steps,
                self._num_bad_steps,
                self._incr_every_n_steps,
                self._decr_every_n_nan_or_inf,
                self._incr_ratio,
                self._decr_ratio,
                name="update_loss_scaling", )

    def apply_gradients(self, params_grads):
        if self._supports_check_nan_inf():
            self._optimizer._set_scale(self._loss_scaling)
            optimize_ops = self._optimizer.apply_gradients(params_grads)
            found_inf = self._optimizer._found_inf
            self._add_dynamic_loss_scaling(params_grads, found_inf)
            return optimize_ops

        found_inf = self._check_finite_and_unscale(params_grads)
        self._add_dynamic_loss_scaling(params_grads, found_inf)

        # Pass found_inf to adam, to skip update for not only param, but also momentum and beta_pow
        # With fleet, optimizers are nested and the real optimizer set by user is the inner most one.
        real_optimizer = self._optimizer
        while hasattr(real_optimizer, "inner_opt"):
            real_optimizer = real_optimizer.inner_opt
        if isinstance(
                real_optimizer,
            (paddle.fluid.optimizer.Adam, paddle.optimizer.AdamW), ):
            # NOTE(zhiqiu): Since found_inf needs to be on cpu in adam op, we
            # copy it in advance to avoid multiple time copies.
            with self._train_program._optimized_guard([]):
                found_inf = paddle.tensor.creation._memcpy(
                    found_inf, paddle.CPUPlace())
            real_optimizer._set_auxiliary_var('found_inf', found_inf)
        elif hasattr(real_optimizer, "_set_auxiliary_var"):
            real_optimizer._set_auxiliary_var('found_inf', found_inf)
        optimize_ops = self._optimizer.apply_gradients(params_grads)
        return optimize_ops


if __name__ == '__main__':
    paddle.enable_static()
    import paddle.fluid as fluid
    place = fluid.CPUPlace()
    main = fluid.Program()
    with fluid.program_guard(main):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)
        adam_optimizer = paddle.optimizer.Adam(1.0e+19)
        #        adam_optimizer.minimize(avg_cost)
        dynamic_loss_scale_opt = LossScaling(adam_optimizer)
        dynamic_loss_scale_opt.minimize(avg_cost)
        print(main)

        fetch_list = [
            avg_cost.name, 'loss_scaling_0', 'find_infinite_scale.tmp_0'
        ]
        #        fetch_list = [avg_cost.name]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=1)
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            out = exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
            print(out)
