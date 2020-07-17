#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import os
import time
import sys
import math
import argparse
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework
import reader
from utils import *
from dy_models import MobileNetV1, MobileNetV2, MobileNetV3_dy, OFA_kernel

args = parse_args()
if int(os.getenv("PADDLE_TRAINER_ID", 0)) == 0:
    print_arguments(args)


def dy_token_gene(ori_token, alter=[3, 5, 7], num_of_changes=1):
    kernel_token = tuple(ori_token[:20])
    exp_token = tuple(ori_token[20:40])
    depth_token = tuple(ori_token[40:45])
    dy_token = []
    for depth_id in range(len(depth_token)):
        for repeat_time in range(depth_token[depth_id]):
            dy_token.append(kernel_token[depth_id * 4 + repeat_time])
    idx_lis = np.random.choice(len(dy_token), num_of_changes, replace=False)
    for idx in idx_lis:
        while True:
            tmp_kernel = alter[np.random.choice(len(alter))]
            if tmp_kernel != dy_token[idx]:
                dy_token[idx] = tmp_kernel
                break
    return dy_token


def dy_token_gene_v2(ori_token, alter=[3, 5, 7], position_from_end=-1):
    kernel_token = tuple(ori_token[:20])
    exp_token = tuple(ori_token[20:40])
    depth_token = tuple(ori_token[40:45])
    dy_token = []
    dy_trainable = []
    for depth_id in range(len(depth_token)):
        for repeat_time in range(depth_token[depth_id]):
            dy_token.append(kernel_token[depth_id * 4 + repeat_time])
            dy_trainable.append(False)

    dy_token[position_from_end] = alter[0]
    dy_trainable[position_from_end] = True

    for i in range(abs(position_from_end) - 1):
        dy_token[position_from_end + i + 1] = alter[np.random.choice(
            len(alter))]
    return dy_token, dy_trainable


def eval(net, test_data_loader, eop, position_from_end=1):
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0
    t_last = 0
    token = [
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3,
        3, 4, 3, 4, 3, 6, 4, 4, 4, 4, 6, 3, 6, 4, 6, 3, 2, 3, 3, 3, 4
    ]
    place_num = paddle.fluid.core.get_cuda_device_count(
    ) if args.use_gpu else int(os.environ.get('CPU_NUM', 1))
    for i, (img, label) in enumerate(test_data_loader()):
        t1 = time.time()
        label = to_variable(label.numpy().astype('int64').reshape(
            int(args.batch_size // place_num), 1))
        if args.model == "once_for_all_kernel":
            dy_token, dy_trainable = dy_token_gene_v2(
                token, alter=[5, 7], position_from_end=position_from_end)
            print(dy_token)
            out = net(img, dy_token=dy_token, dy_trainable=dy_trainable)
        else:
            out = net(img)
        softmax_out = fluid.layers.softmax(out, use_cudnn=False)
        loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
        avg_loss = fluid.layers.mean(x=loss)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
        t2 = time.time()
        print( "test | epoch id: %d, batch id: %d, avg_loss %0.5f acc_top1 %0.5f acc_top5 %0.5f %2.4f sec read_t:%2.4f" % \
                (eop, i, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy(), t2 - t1 , t1 - t_last))
        sys.stdout.flush()
        total_loss += avg_loss.numpy()
        total_acc1 += acc_top1.numpy()
        total_acc5 += acc_top5.numpy()
        total_sample += 1
        t_last = time.time()
    print("final eval loss %0.3f acc1 %0.3f acc5 %0.3f" % \
          (total_loss / total_sample, \
           total_acc1 / total_sample, total_acc5 / total_sample))
    sys.stdout.flush()


def train_mobilenet():
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        # 1. init net and optimizer
        place_num = paddle.fluid.core.get_cuda_device_count(
        ) if args.use_gpu else int(os.environ.get('CPU_NUM', 1))
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        if args.model == "MobileNetV1":
            net = MobileNetV1(class_dim=args.class_dim, scale=1.0)
            model_path_pre = 'mobilenet_v1'
        elif args.model == "MobileNetV2":
            net = MobileNetV2(class_dim=args.class_dim, scale=1.0)
            model_path_pre = 'mobilenet_v2'
        elif args.model == "MobileNetV3":
            net = MobileNetV3_dy(
                class_dim=args.class_dim, scale=1.0, model_name='large')
            model_path_pre = 'mobilenet_v3'
        elif args.model == "once_for_all":
            token = [
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3,
                3, 3, 3, 3, 4, 3, 4, 3, 6, 4, 4, 4, 4, 6, 3, 6, 4, 6, 3, 2, 3,
                3, 3, 4
            ]
            net = OFA_kernel(
                class_dim=args.class_dim,
                scale=1.0,
                model_name='large',
                token=token,
                ofa_mode=False)
            model_path_pre = 'ofa'
        elif args.model == "once_for_all_kernel":
            token = [
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3,
                3, 3, 3, 3, 4, 3, 4, 3, 6, 4, 4, 4, 4, 6, 3, 6, 4, 6, 3, 2, 3,
                3, 3, 4
            ]
            net = OFA_kernel(
                class_dim=args.class_dim,
                scale=1.0,
                model_name='large',
                token=token,
                ofa_mode='kernel',
                trainable_besides_trans=False)
            model_path_pre = 'ofa_kernel'
        else:
            print(
                "wrong model name, please try model = MobileNetV1 or MobileNetV2"
            )
            exit()

        optimizer = create_optimizer(args=args, parameter_list=net.parameters())
        if args.use_data_parallel:
            net = fluid.dygraph.parallel.DataParallel(net, strategy)

        # 2. load checkpoint
        if args.checkpoint:
            assert os.path.exists(args.checkpoint + ".pdparams"), \
                "Given dir {}.pdparams not exist.".format(args.checkpoint)
            assert os.path.exists(args.checkpoint + ".pdopt"), \
                "Given dir {}.pdopt not exist.".format(args.checkpoint)
            para_dict, opti_dict = fluid.dygraph.load_dygraph(args.checkpoint)

            if args.model == "once_for_all_kernel":
                inner_state_dict = net.state_dict()
                for name, para in inner_state_dict.items():
                    key_name = name
                    if key_name in para_dict:
                        print(key_name)
                        para.set_value(para_dict[key_name])
            else:
                net.set_dict(para_dict)
                optimizer.set_dict(opti_dict)

        # 3. reader
        train_data_loader, train_data = utility.create_data_loader(
            is_train=True, args=args)
        test_data_loader, test_data = utility.create_data_loader(
            is_train=False, args=args)
        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
        imagenet_reader = reader.ImageNetReader(seed=0, place_num=place_num)
        train_reader = imagenet_reader.train(settings=args)
        test_reader = imagenet_reader.val(settings=args)
        train_data_loader.set_sample_list_generator(train_reader, place)
        test_data_loader.set_sample_list_generator(test_reader, place)
        with open('./dy_token.txt', 'r+') as f:
            lines = f.readlines()

        eval_sign = False
        if eval_sign:
            net.eval()
            eval(net, test_data_loader, 1, -1)
            exit()

        # 4. train loop
        total_batch_num = 0  #this is for benchmark
        # Hyper-Params
        total_epoch_each_layer = 15
        fixed_epoch_each_layer = 10
        position = 0
        count = 0
        for eop in range(args.num_epochs):
            if num_trainers > 1:
                imagenet_reader.set_shuffle_seed(eop + (
                    args.random_seed if args.random_seed else 0))
            net.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0
            batch_id = 0
            t_last = 0

            if eop % total_epoch_each_layer == 0:
                position += 1
                count = 0
                with open(
                        os.path.join('./token', str(position) + '.txt'),
                        'r+') as f:
                    lines = f.readlines()
            count += 1

            # 4.1 for each batch, call net() , backward(), and minimize()
            for idx, (img, label) in enumerate(train_data_loader()):
                t1 = time.time()
                if args.max_iter and total_batch_num == args.max_iter:
                    return
                label = to_variable(label.numpy().astype('int64').reshape(
                    int(args.batch_size // place_num), 1))
                t_start = time.time()

                # 4.1.1 call net()
                if args.model == "once_for_all_kernel":
                    dy_token = [int(x) for x in lines[idx].strip().split(',')]
                    if count <= fixed_epoch_each_layer:
                        dy_trainable = [False for _ in range(len(dy_token))]
                        dy_trainable[0 - position] = True
                    else:
                        dy_trainable = [False] * (len(dy_token) - position
                                                  ) + [True] * position
                    if idx < 10:
                        print(dy_token)
                        print(dy_trainable)
                    out = net(img, dy_token=dy_token, dy_trainable=dy_trainable)
                else:
                    out = net(img)

                t_end = time.time()
                softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                loss = fluid.layers.cross_entropy(
                    input=softmax_out, label=label)
                avg_loss = fluid.layers.mean(x=loss)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
                t_start_back = time.time()

                # 4.1.2 call backward()
                if args.use_data_parallel:
                    avg_loss = net.scale_loss(avg_loss)
                    avg_loss.backward()
                    net.apply_collective_grads()
                else:
                    avg_loss.backward()

                t_end_back = time.time()

                # 4.1.3 call minimize()
                optimizer.minimize(avg_loss)

                net.clear_gradients()
                t2 = time.time()
                train_batch_elapse = t2 - t1
                if batch_id % args.print_step == 0:
                    print( "epoch id: %d, batch step: %d, avg_loss %0.5f, acc_top1 %0.5f, acc_top5 %0.5f, lr %f, %2.4f sec, net_t:%2.4f, back_t:%2.4f, read_t:%2.4f" % \
                            (eop, batch_id, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy(), optimizer._global_learning_rate().numpy(), train_batch_elapse,
                              t_end - t_start, t_end_back - t_start_back,  t1 - t_last))
                    sys.stdout.flush()
                total_loss += avg_loss.numpy()
                total_acc1 += acc_top1.numpy()
                total_acc5 += acc_top5.numpy()
                total_sample += 1
                batch_id += 1
                t_last = time.time()

                # NOTE: used for benchmark
                total_batch_num = total_batch_num + 1

            if args.ce:
                print("kpis\ttrain_acc1\t%0.3f" % (total_acc1 / total_sample))
                print("kpis\ttrain_acc5\t%0.3f" % (total_acc5 / total_sample))
                print("kpis\ttrain_loss\t%0.3f" % (total_loss / total_sample))
            print("epoch %d | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f %2.4f sec" % \
                  (eop, batch_id, total_loss / total_sample, \
                   total_acc1 / total_sample, total_acc5 / total_sample, train_batch_elapse))

            # 4.2 save checkpoint
            save_parameters = (not args.use_data_parallel) or (
                args.use_data_parallel and
                fluid.dygraph.parallel.Env().local_rank == 0)
            if save_parameters:
                if not os.path.isdir(args.model_save_dir):
                    os.makedirs(args.model_save_dir)
                model_path = os.path.join(
                    args.model_save_dir,
                    "_" + model_path_pre + "_epoch{}".format(eop))
                fluid.dygraph.save_dygraph(net.state_dict(), model_path)
                fluid.dygraph.save_dygraph(optimizer.state_dict(), model_path)

            # 4.3 validation
            net.eval()
            eval(net, test_data_loader, eop, 0 - position)

        # 5. save final results
        save_parameters = (not args.use_data_parallel) or (
            args.use_data_parallel and
            fluid.dygraph.parallel.Env().local_rank == 0)
        if save_parameters:
            model_path = os.path.join(args.model_save_dir,
                                      "_" + model_path_pre + "_final")
            fluid.dygraph.save_dygraph(net.state_dict(), model_path)


if __name__ == '__main__':

    train_mobilenet()
