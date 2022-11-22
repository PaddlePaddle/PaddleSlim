from __future__ import absolute_import
import paddle
from ..models import classification_models

__all__ = ["image_classification"]

model_list = classification_models.model_list


def image_classification(model, image_shape, class_num, use_gpu=False):
    assert model in model_list
    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        image = paddle.static.data(
            name='image', shape=image_shape, dtype='float32')
        label = paddle.static.data(name='label', shape=[1], dtype='int64')
        model = classification_models.__dict__[model]()
        out = model.net(input=image, class_dim=class_num)
        cost = paddle.nn.functional.cross_entropy(input=out, label=label)
        avg_cost = paddle.mean(x=cost)
        acc_top1 = paddle.static.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.static.accuracy(input=out, label=label, k=5)
        val_program = paddle.static.default_main_program().clone(for_test=True)

        opt = paddle.optimizer.Momentum(learning_rate=0.1, momentum=0.9)
        opt.minimize(avg_cost)
        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
    return exe, train_program, val_program, (image, label), (
        acc_top1.name, acc_top5.name, avg_cost.name, out.name)
