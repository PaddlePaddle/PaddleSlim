from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import functools
import paddle
import paddle.fluid as fluid
import argparse
import subprocess
import sys
sys.path.append('../..')
sys.path.append('.')

import imagenet_reader as reader
import models
from utility import add_arguments, print_arguments
from utility import save_persistable_nodes, load_persistable_nodes
import quant


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   256,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   1281167,              "Training image number.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('model_save_dir',   str,   "output",             "model save directory")
add_arg('pretrained_fp32_model', str,   None,            "Whether to use the pretrained float32 model to initialize the weights.")
add_arg('checkpoint',       str,   None,                 "Whether to resume the training process from the checkpoint.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
add_arg('act_quant_type',   str,   "abs_max",            "quantization type for activation, valid value:'abs_max','range_abs_max', 'moving_average_abs_max'" )
add_arg('wt_quant_type',    str,   "abs_max",            "quantization type for weight, valid value:'abs_max','channel_wise_abs_max'" )
# yapf: enabl


def optimizer_setting(params):
    ls = params["learning_strategy"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        print("decay list:{}".format(bd))
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(4e-5))
    elif ls["name"] == "exponential_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size +1)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        learning_decay_rate_factor=ls["learning_decay_rate_factor"]
        num_epochs_per_decay = ls["num_epochs_per_decay"]
        NUM_GPUS = 1

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate = lr * NUM_GPUS,
                decay_steps = step * num_epochs_per_decay / NUM_GPUS,
                decay_rate = learning_decay_rate_factor),
            momentum=0.9,

            regularization = fluid.regularizer.L2Decay(4e-5))

    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer

def net_config(image, label, model, args):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list,"{} is not lists: {}".format(
        args.model, model_list)

    class_dim = args.class_dim
    model_name = args.model

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
        out = out0
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    return out, avg_cost, acc_top1, acc_top5


def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    model = models.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            out, avg_cost, acc_top1, acc_top5 = net_config(image, label, model, args)
            avg_cost.persistable = True
            acc_top1.persistable = True
            acc_top5.persistable = True
            if is_train:
                params = model.params
                params["total_images"] = args.total_images
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["learning_strategy"]["batch_size"] = args.batch_size
                params["learning_strategy"]["name"] = args.lr_strategy

                optimizer = optimizer_setting(params)
                optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()
    if is_train:
        return image, out, py_reader, avg_cost, acc_top1, acc_top5, global_lr
    else:
        return image, out, py_reader, avg_cost, acc_top1, acc_top5

def train(args):
    ############################################################################################################
    # 1. quantization configs
    ############################################################################################################
    quant_config = {
        # weight quantize type, default is 'abs_max'
        'weight_quantize_type': 'abs_max',
        # activation quantize type, default is 'abs_max'
        'activation_quantize_type': 'abs_max',
        # weight quantize bit num, default is 8
        'weight_bits': 8,
        # activation quantize bit num, default is 8
        'activation_bits': 8,
        # op of name_scope in not_quant_pattern list, will not quantized
        'not_quant_pattern': ['skip_quant'],
        # op of types in quantize_op_types, will quantized
        'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
        # data type after quantization, default is 'int8'
        'dtype': 'int8',
        # window size for 'range_abs_max' quantization. defaulf is 10000
        'window_size': 10000,
        # The decay coefficient of moving average, default is 0.9
        'moving_rate': 0.9,
        # if set quant_weight_only True, then only quantize parameters of layers which need quantization,
        # and insert anti-quantization op for parameters of these layers.
        'quant_weight_only': False
    }


    # parameters from arguments
    model_name = args.model
    pretrained_fp32_model = args.pretrained_fp32_model
    checkpoint = args.checkpoint
    model_save_dir = args.model_save_dir
    data_dir = args.data_dir
    activation_quant_type = args.act_quant_type
    weight_quant_type = args.wt_quant_type
    print("Using %s as the actiavtion quantize type." % activation_quant_type)
    print("Using %s as the weight quantize type." % weight_quant_type)

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    _, _, train_py_reader, train_cost, train_acc1, train_acc5, global_lr = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    image, out, test_py_reader, test_cost, test_acc1, test_acc5 = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = test_prog.clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    scope = fluid.global_scope()
    exe = fluid.Executor(place)

    ############################################################################################################
    # 2. quantization transform programs (training aware)
    #    Make some quantization transforms in the graph before training and testing.
    #    According to the weight and activation quantization type, the graph will be added
    #    some fake quantize operators and fake dequantize operators.
    ############################################################################################################
    train_prog = quant.quanter.quant_aware(train_prog, scope, place, quant_config, for_test = False)
    test_prog = quant.quanter.quant_aware(test_prog, scope, place, quant_config, for_test=True)

    # load checkpoint todo


    exe.run(startup_prog)

    if pretrained_fp32_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_fp32_model, var.name))
        fluid.io.load_vars(
            exe, pretrained_fp32_model, main_program=train_prog, predicate=if_exist)

    if args.use_gpu:
        visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
        if visible_device:
            device_num = len(visible_device.split(','))
        else:
            device_num = subprocess.check_output(
                ['nvidia-smi', '-L']).decode().count('\n')
    else:
        device_num = 1

    train_batch_size = args.batch_size / device_num
    test_batch_size = 1 if activation_quant_type == 'abs_max' else 8
    train_reader = paddle.batch(
        reader.train(data_dir=data_dir), batch_size=train_batch_size, drop_last=True)
    test_reader = paddle.batch(reader.val(data_dir=data_dir), batch_size=test_batch_size)

    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)

    train_fetch_list = [train_cost.name, train_acc1.name, train_acc5.name, global_lr.name]
    test_fetch_list = [test_cost.name, test_acc1.name, test_acc5.name]


    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = False
    params = models.__dict__[args.model]().params
    for pass_id in range(params["num_epochs"]):

        train_py_reader.start()

        train_info = [[], [], []]
        test_info = [[], [], []]
        train_time = []
        batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc1, acc5, lr = exe.run(train_prog, fetch_list=train_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(np.array(loss))
                acc1 = np.mean(np.array(acc1))
                acc5 = np.mean(np.array(acc5))
                train_info[0].append(loss)
                train_info[1].append(acc1)
                train_info[2].append(acc5)
                lr = np.mean(np.array(lr))
                train_time.append(period)
                if batch_id % 10 == 0:
                    print("Pass {0}, trainbatch {1}, loss {2}, \
                        acc1 {3}, acc5 {4}, lr {5}, time {6}"
                          .format(pass_id, batch_id, loss, acc1, acc5, "%.6f" %
                                  lr, "%2.2f sec" % period))
                    sys.stdout.flush()
                batch_id += 1
        except fluid.core.EOFException:
            train_py_reader.reset()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()

        test_py_reader.start()

        test_batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc1, acc5 = exe.run(program=test_prog,
                                           fetch_list=test_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(loss)
                acc1 = np.mean(acc1)
                acc5 = np.mean(acc5)
                test_info[0].append(loss)
                test_info[1].append(acc1)
                test_info[2].append(acc5)
                if test_batch_id % 10 == 0:
                    print("Pass {0},testbatch {1},loss {2}, \
                        acc1 {3},acc5 {4},time {5}"
                          .format(pass_id, test_batch_id, loss, acc1, acc5,
                                  "%2.2f sec" % period))
                    sys.stdout.flush()
                test_batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, "
              "test_loss {4}, test_acc1 {5}, test_acc5 {6}".format(
                  pass_id, train_loss, train_acc1, train_acc5, test_loss,
                  test_acc1, test_acc5))
        sys.stdout.flush()

        # save checkpoints todo
        # save_checkpoint_path = os.path.join(model_save_dir,  model_name, str(pass_id))
        # if not os.path.isdir(save_checkpoint_path):
        #     os.makedirs(save_checkpoint_path)
        # save_persistable_nodes(exe, save_checkpoint_path, main_graph)




    ############################################################################################################
    # 3. Freeze the graph after training by adjusting the quantize
    #    operators' order for the inference.
    #    The dtype of float_program's weights is float32, but in int8 range.
    ############################################################################################################
    float_program, int8_program = quant.quanter.convert(test_prog, scope, place, quant_config, save_int8=True)


    ############################################################################################################
    # 4. Save inference model
    ############################################################################################################
    model_path = os.path.join(model_save_dir, model_name, args.act_quant_type)
    float_path = os.path.join(model_path, 'float')
    int8_path = os.path.join(model_path, 'int8')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    fluid.io.save_inference_model(
        dirname=float_path,
        feeded_var_names=[image.name],
        target_vars=[out], executor=exe,
        main_program=float_program)

    fluid.io.save_inference_model(
        dirname=int8_path,
        feeded_var_names=[image.name],
        target_vars=[out], executor=exe,
        main_program=int8_program)



def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
