import numpy as np
from itertools import izip
import paddle.fluid as fluid
from paddleslim.teachers.bert.reader.cls import *
from paddleslim.nas.darts.search_space import AdaBERTClassifier
from paddle.fluid.dygraph.base import to_variable
from tqdm import tqdm
import os
import pickle

import logging
from paddleslim.common import AvgrageMeter, get_logger
logger = get_logger(__name__, level=logging.INFO)


def valid_one_epoch(model, valid_loader, epoch, log_freq):
    accs = AvgrageMeter()
    ce_losses = AvgrageMeter()
    model.student.eval()

    step_id = 0
    for valid_data in valid_loader():
        try:
            loss, acc, ce_loss, _, _ = model._layers.loss(valid_data, epoch)
        except:
            loss, acc, ce_loss, _, _ = model.loss(valid_data, epoch)

        batch_size = valid_data[0].shape[0]
        ce_losses.update(ce_loss.numpy(), batch_size)
        accs.update(acc.numpy(), batch_size)
        step_id += 1
    return ce_losses.avg[0], accs.avg[0]


def train_one_epoch(model, train_loader, optimizer, epoch, use_data_parallel,
                    log_freq):
    total_losses = AvgrageMeter()
    accs = AvgrageMeter()
    ce_losses = AvgrageMeter()
    kd_losses = AvgrageMeter()
    model.student.train()

    step_id = 0
    for train_data in train_loader():
        batch_size = train_data[0].shape[0]

        if use_data_parallel:
            total_loss, acc, ce_loss, kd_loss, _ = model._layers.loss(
                train_data, epoch)
        else:
            total_loss, acc, ce_loss, kd_loss, _ = model.loss(train_data,
                                                              epoch)

        if use_data_parallel:
            total_loss = model.scale_loss(total_loss)
            total_loss.backward()
            model.apply_collective_grads()
        else:
            total_loss.backward()
        optimizer.minimize(total_loss)
        model.clear_gradients()
        total_losses.update(total_loss.numpy(), batch_size)
        accs.update(acc.numpy(), batch_size)
        ce_losses.update(ce_loss.numpy(), batch_size)
        kd_losses.update(kd_loss.numpy(), batch_size)

        if step_id % log_freq == 0:
            logger.info(
                "Train Epoch {}, Step {}, Lr {:.6f} total_loss {:.6f}; ce_loss {:.6f}, kd_loss {:.6f}, train_acc {:.6f};".
                format(epoch, step_id,
                       optimizer.current_step_lr(), total_losses.avg[0],
                       ce_losses.avg[0], kd_losses.avg[0], accs.avg[0]))
        step_id += 1


def main():
    # whether use multi-gpus
    use_data_parallel = False
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env(
    ).dev_id) if use_data_parallel else fluid.CUDAPlace(0)

    BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12"
    vocab_path = BERT_BASE_PATH + "/vocab.txt"

    do_lower_case = True
    # augmented dataset nums
    # num_samples = 8016987

    max_seq_len = 128
    batch_size = 192
    hidden_size = 768
    emb_size = 768
    epoch = 80
    log_freq = 10

    task_name = 'mnli'

    if task_name == 'mrpc':
        data_dir = "./data/glue_data/MRPC/"
        teacher_model_dir = "./data/teacher_model/mrpc"
        num_samples = 3668
        max_layer = 4
        num_labels = 2
        processor_func = MrpcProcessor
    elif task_name == 'mnli':
        data_dir = "./data/glue_data/MNLI/"
        teacher_model_dir = "./data/teacher_model/steps_23000"
        num_samples = 392702
        max_layer = 8
        num_labels = 3
        processor_func = MnliProcessor

    device_num = fluid.dygraph.parallel.Env().nranks
    use_fixed_gumbel = True
    train_phase = "train"
    val_phase = "dev"
    step_per_epoch = int(num_samples / (batch_size * device_num))

    with fluid.dygraph.guard(place):
        if use_fixed_gumbel:
            # make sure gumbel arch is constant
            np.random.seed(1)
            fluid.default_main_program().random_seed = 1
        model = AdaBERTClassifier(
            num_labels,
            n_layer=max_layer,
            hidden_size=hidden_size,
            task_name=task_name,
            emb_size=emb_size,
            teacher_model=teacher_model_dir,
            data_dir=data_dir,
            use_fixed_gumbel=use_fixed_gumbel)

        learning_rate = fluid.dygraph.CosineDecay(2e-2, step_per_epoch, epoch)

        model_parameters = []
        for p in model.parameters():
            if (p.name not in [a.name for a in model.arch_parameters()] and
                    p.name not in
                [a.name for a in model.teacher.parameters()]):
                model_parameters.append(p)

        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(3e-4),
            parameter_list=model_parameters)

        processor = processor_func(
            data_dir=data_dir,
            vocab_path=vocab_path,
            max_seq_len=max_seq_len,
            do_lower_case=do_lower_case,
            in_tokens=False)

        train_reader = processor.data_generator(
            batch_size=batch_size,
            phase=train_phase,
            epoch=1,
            dev_count=1,
            shuffle=True)
        dev_reader = processor.data_generator(
            batch_size=batch_size,
            phase=val_phase,
            epoch=1,
            dev_count=1,
            shuffle=False)

        if use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=128,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        dev_loader = fluid.io.DataLoader.from_generator(
            capacity=128,
            use_double_buffer=True,
            iterable=True,
            return_list=True)

        train_loader.set_batch_generator(train_reader, places=place)
        dev_loader.set_batch_generator(dev_reader, places=place)

        if use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        best_valid_acc = 0
        for epoch_id in range(epoch):
            train_one_epoch(model, train_loader, optimizer, epoch_id,
                            use_data_parallel, log_freq)
            loss, acc = valid_one_epoch(model, dev_loader, epoch_id, log_freq)
            if acc > best_valid_acc:
                best_valid_acc = acc
            logger.info(
                "dev set, ce_loss {:.6f}; acc {:.6f}, best_acc {:.6f};".format(
                    loss, acc, best_valid_acc))


if __name__ == '__main__':
    main()
