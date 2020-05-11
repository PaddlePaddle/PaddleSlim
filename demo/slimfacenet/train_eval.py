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
import shutil
import subprocess
import argparse
import time
import scipy.io
import numpy as np

import paddle
from paddle import fluid

from dataloader.casia import CASIA_Face
from dataloader.lfw import LFW
from lfw_eval import parse_filelist, evaluation_10_fold
from models.slimfacenet import SlimFaceNet


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def creat_optimizer(args, trainset_scale):
    start_step = trainset_scale * args.start_epoch // args.train_batchsize

    if args.lr_strategy == 'piecewise_decay':
        bd = [
            trainset_scale * int(e) // args.train_batchsize
            for e in args.lr_steps.strip().split(',')
        ]
        lr = [float(e) for e in args.lr_list.strip().split(',')]
        assert len(bd) == len(lr) - 1
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(args.l2_decay))
    elif args.lr_strategy == 'cosine_decay':
        lr = args.lr
        step_each_epoch = trainset_scale // args.train_batchsize
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.cosine_decay(lr, step_each_epoch,
                                                    args.total_epoch),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(args.l2_decay))
    else:
        print('Wrong learning rate strategy')
        exit()
    return optimizer


def test(test_exe, test_program, test_out, args):
    featureLs = None
    featureRs = None
    out_feature, test_reader, flods, flags = test_out
    for idx, data in enumerate(test_reader()):
        res = []
        res.append(
            test_exe.run(test_program,
                         feed={u'image_test': data[0][u'image_test1']},
                         fetch_list=out_feature))
        res.append(
            test_exe.run(test_program,
                         feed={u'image_test': data[0][u'image_test2']},
                         fetch_list=out_feature))
        res.append(
            test_exe.run(test_program,
                         feed={u'image_test': data[0][u'image_test3']},
                         fetch_list=out_feature))
        res.append(
            test_exe.run(test_program,
                         feed={u'image_test': data[0][u'image_test4']},
                         fetch_list=out_feature))
        featureL = np.concatenate((res[0][0], res[1][0]), 1)
        featureR = np.concatenate((res[2][0], res[3][0]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(args.feature_save_dir, result)
    ACCs = evaluation_10_fold(args.feature_save_dir)
    print('eval arch {}'.format(args.arch))
    with open(os.path.join(args.save_ckpt, 'log.txt'), 'a+') as f:
        f.writelines('eval arch {}\n'.format(args.arch))
    for i in range(len(ACCs)):
        print('{}    {}'.format(i + 1, ACCs[i] * 100))
        with open(os.path.join(args.save_ckpt, 'log.txt'), 'a+') as f:
            f.writelines('{}    {}\n'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('AVE {}'.format(np.mean(ACCs) * 100))
    with open(os.path.join(args.save_ckpt, 'log.txt'), 'a+') as f:
        f.writelines('--------\n')
        f.writelines('AVE    {}\n'.format(np.mean(ACCs) * 100))
    return np.mean(ACCs) * 100


def train(exe, train_program, train_out, test_program, test_out, args):
    loss, acc, global_lr, train_reader = train_out
    fetch_list_train = [loss.name, acc.name, global_lr.name]
    train_exe = fluid.ParallelExecutor(
        use_cuda=True, loss_name=loss.name, main_program=train_program)
    for epoch_id in range(args.start_epoch, args.total_epoch):
        for batch_id, data in enumerate(train_reader()):
            loss, acc, global_lr = train_exe.run(feed=data,
                                                 fetch_list=fetch_list_train)
            avg_loss = np.mean(np.array(loss))
            avg_acc = np.mean(np.array(acc))
            print(
                '{}  Epoch: {:^4d} step: {:^4d} loss: {:.6f}, acc: {:.6f}, lr: {}'.
                format(now(), epoch_id, batch_id, avg_loss, avg_acc,
                       float(np.mean(np.array(global_lr)))))

        if batch_id % args.save_frequency == 0:
            model_path = os.path.join(args.save_ckpt, str(epoch_id))
            fluid.io.save_persistables(
                executor=exe, dirname=model_path, main_program=train_program)

            test(exe, test_program, test_out, args)


def build_program(program, startup, args, is_train=True):
    num_trainers = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    places = fluid.cuda_places() if args.use_gpu else fluid.CPUPlace()

    train_dataset = CASIA_Face(root=args.train_data_dir)
    trainset_scale = len(train_dataset)

    with fluid.program_guard(main_program=program, startup_program=startup):
        with fluid.unique_name.guard():
            # Model construction
            arch = [int(a) for a in args.arch.strip().split(',')]
            model = SlimFaceNet(
                class_dim=train_dataset.class_nums,
                scale=args.scale,
                arch=arch)

            if is_train:
                image = fluid.layers.data(
                    name='image', shape=[-1, 3, 112, 112], dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[-1, 1], dtype='int64')
                train_reader = paddle.batch(
                    train_dataset.reader,
                    batch_size=args.train_batchsize // num_trainers,
                    drop_last=False)
                reader = fluid.io.PyReader(
                    feed_list=[image, label],
                    capacity=64,
                    iterable=True,
                    return_list=False)
                reader.decorate_sample_list_generator(
                    train_reader, places=places)

                model.extract_feature = False
                loss, acc = model.net(image, label)
                optimizer = creat_optimizer(args, trainset_scale)
                optimizer.minimize(loss)
                global_lr = optimizer._global_learning_rate()
                out = (loss, acc, global_lr, reader)

            else:
                nl, nr, flods, flags = parse_filelist(args.test_data_dir)
                test_dataset = LFW(nl, nr)
                test_reader = paddle.batch(
                    test_dataset.reader,
                    batch_size=args.test_batchsize,
                    drop_last=False)
                image_test = fluid.layers.data(
                    name='image_test',
                    shape=[-1, 3, 112, 112],
                    dtype='float32')
                image_test1 = fluid.layers.data(
                    name='image_test1',
                    shape=[-1, 3, 112, 112],
                    dtype='float32')
                image_test2 = fluid.layers.data(
                    name='image_test2',
                    shape=[-1, 3, 112, 112],
                    dtype='float32')
                image_test3 = fluid.layers.data(
                    name='image_test3',
                    shape=[-1, 3, 112, 112],
                    dtype='float32')
                image_test4 = fluid.layers.data(
                    name='image_test4',
                    shape=[-1, 3, 112, 112],
                    dtype='float32')
                reader = fluid.io.PyReader(
                    feed_list=[
                        image_test1, image_test2, image_test3, image_test4
                    ],
                    capacity=64,
                    iterable=True,
                    return_list=False)
                reader.decorate_sample_list_generator(test_reader,
                                                      fluid.core.CPUPlace())

                model.extract_feature = True
                feature = model.net(image_test)
                out = (feature, reader, flods, flags)

            return out


def main():
    global args
    parser = argparse.ArgumentParser(description='PaddlePaddle SlimFaceNet')
    parser.add_argument(
        '--action', default='final', type=str, help='test/final')
    parser.add_argument(
        '--model',
        default='slimfacenet',
        type=str,
        help='slimfacenet/slimfacenet_v1')
    parser.add_argument('--scale', default=0.6, type=float, help='scale ratio')
    parser.add_argument(
        '--arch',
        default='1,1,0,1,1,1,1,0,1,0,1,3,2,2,3',
        type=str,
        help='arch')
    parser.add_argument(
        '--use_gpu', default=1, type=int, help='Use GPU or not, 0 is not used')
    parser.add_argument(
        '--use_multiGPU',
        default=0,
        type=int,
        help='Use multi GPU or not, 0 is not used')
    parser.add_argument(
        '--lr_strategy',
        default='piecewise_decay',
        type=str,
        help='lr_strategy')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument(
        '--lr_list',
        default='0.1,0.01,0.001,0.0001',
        type=str,
        help='learning rate list (piecewise_decay)')
    parser.add_argument(
        '--lr_steps',
        default='36,52,58',
        type=str,
        help='learning rate decay at which epochs')
    parser.add_argument(
        '--l2_decay', default=4e-5, type=float, help='base l2_decay')
    parser.add_argument(
        '--train_data_dir', default='./CASIA', type=str, help='train_data_dir')
    parser.add_argument(
        '--test_data_dir', default='./lfw', type=str, help='lfw_data_dir')
    parser.add_argument(
        '--train_batchsize', default=512, type=int, help='train_batchsize')
    parser.add_argument(
        '--test_batchsize', default=500, type=int, help='test_batchsize')
    parser.add_argument(
        '--img_shape', default='3,112,96', type=str, help='img_shape')
    parser.add_argument(
        '--start_epoch', default=0, type=int, help='start_epoch')
    parser.add_argument(
        '--total_epoch', default=80, type=int, help='total_epoch')
    parser.add_argument(
        '--save_frequency', default=1, type=int, help='save_frequency')
    parser.add_argument(
        '--save_ckpt', default='output', type=str, help='save_ckpt')
    parser.add_argument('--resume', default='', type=str, help='resume')
    parser.add_argument(
        '--feature_save_dir',
        default='result.mat',
        type=str,
        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()

    num_trainers = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    print(args)
    print('num_trainers: {}'.format(num_trainers))
    if args.save_ckpt == None:
        args.save_ckpt = 'output'
    if not os.path.exists(args.save_ckpt):
        subprocess.call(['mkdir', '-p', args.save_ckpt])
    shutil.copyfile(__file__, os.path.join(args.save_ckpt, 'train.py'))
    shutil.copyfile('models/slimfacenet.py',
                    os.path.join(args.save_ckpt, 'model.py'))
    with open(os.path.join(args.save_ckpt, 'log.txt'), 'w+') as f:
        f.writelines(str(args) + '\n')
        f.writelines('num_trainers: {}'.format(num_trainers) + '\n')

    if args.action == 'final':
        train_program = fluid.Program()
    test_program = fluid.Program()
    startup_program = fluid.Program()

    if args.action == 'final':
        train_out = build_program(train_program, startup_program, args, True)
    test_out = build_program(test_program, startup_program, args, False)
    test_program = test_program.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    if args.action == 'final':
        train(exe, train_program, train_out, test_program, test_out, args)
    if args.action == 'test':
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             dirname='./quant_model/',
             model_filename=None,
             params_filename=None,
             executor=exe)
        nl, nr, flods, flags = parse_filelist(args.test_data_dir)
        test_dataset = LFW(nl, nr)
        test_reader = paddle.batch(
            test_dataset.reader,
            batch_size=args.test_batchsize,
            drop_last=False)
        image_test = fluid.layers.data(
            name='image_test', shape=[-1, 3, 112, 96], dtype='float32')
        image_test1 = fluid.layers.data(
            name='image_test1', shape=[-1, 3, 112, 96], dtype='float32')
        image_test2 = fluid.layers.data(
            name='image_test2', shape=[-1, 3, 112, 96], dtype='float32')
        image_test3 = fluid.layers.data(
            name='image_test3', shape=[-1, 3, 112, 96], dtype='float32')
        image_test4 = fluid.layers.data(
            name='image_test4', shape=[-1, 3, 112, 96], dtype='float32')
        reader = fluid.io.PyReader(
            feed_list=[image_test1, image_test2, image_test3, image_test4],
            capacity=64,
            iterable=True,
            return_list=False)
        reader.decorate_sample_list_generator(test_reader,
                                              fluid.core.CPUPlace())
        test_out = (fetch_targets, reader, flods, flags)
        print('fetch_targets[0]: ', fetch_targets[0])
        print('feed_target_names: ', feed_target_names)
        test(exe, inference_program, test_out, args)
    else:
        print('WRONG ACTION')


if __name__ == '__main__':
    main()
