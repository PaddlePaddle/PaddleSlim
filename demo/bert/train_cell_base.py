import numpy as np
from itertools import izip
import paddle.fluid as fluid
from paddleslim.teachers.bert.reader.cls import *
from paddleslim.nas.darts.search_space import AdaBERTClassifier
from paddleslim.nas.darts.architect_for_bert import Architect

import logging
from paddleslim.common import AvgrageMeter, get_logger
logger = get_logger(__name__, level=logging.INFO)


def count_parameters_in_MB(all_params):
    parameters_number = 0
    for param in all_params:
        if param.trainable:
            parameters_number += np.prod(param.shape)
    return parameters_number / 1e6


def model_loss(model, data_ids):
    # src_ids = data_ids[0]
    # position_ids = data_ids[1]
    # sentence_ids = data_ids[2]
    # input_mask = data_ids[3]
    labels = data_ids[4]
    labels.stop_gradient = True

    enc_output = model(data_ids)

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=enc_output, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)
    return loss, accuracy


def train_one_epoch(model, architect, train_loader, valid_loader, optimizer,
                    epoch, use_data_parallel, log_freq):
    ce_losses = AvgrageMeter()
    accs = AvgrageMeter()
    model.train()

    step_id = 0
    for train_data, valid_data in izip(train_loader(), valid_loader):
        #        architect.step(train_data, valid_data)
        loss, acc = model_loss(model, train_data)

        if use_data_parallel:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()

        optimizer.minimize(loss)
        model.clear_gradients()

        batch_size = train_data[0].shape[0]
        ce_losses.update(loss.numpy(), batch_size)
        accs.update(acc.numpy(), batch_size)

        if step_id % log_freq == 0:
            logger.info(
                "Train Epoch {}, Step {}, Lr {:.6f} loss {:.6f}; acc: {:.6f};".
                format(epoch, step_id,
                       optimizer.current_step_lr(), ce_losses.avg[0], accs.avg[
                           0]))
        step_id += 1


def valid_one_epoch(model, valid_loader, epoch, log_freq):
    ce_losses = AvgrageMeter()
    accs = AvgrageMeter()
    model.eval()

    step_id = 0
    for valid_data in valid_loader():
        loss, acc = model_loss(model, valid_data)

        batch_size = valid_data[0].shape[0]
        ce_losses.update(loss.numpy(), batch_size)
        accs.update(acc.numpy(), batch_size)

        if step_id % log_freq == 0:
            logger.info("Valid Epoch {}, Step {}, loss {:.6f}; acc: {:.6f};".
                        format(epoch, step_id, ce_losses.avg[0], accs.avg[0]))
        step_id += 1


def main():
    use_data_parallel = False
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env(
    ).dev_id) if use_data_parallel else fluid.CUDAPlace(0)

    BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12"
    bert_config_path = BERT_BASE_PATH + "/bert_config.json"
    vocab_path = BERT_BASE_PATH + "/vocab.txt"
    data_dir = "./data/glue_data/MNLI/"
    teacher_model_dir = "./teacher_model/steps_23000"
    num_samples = 392702
    max_seq_len = 128
    do_lower_case = True
    batch_size = 128
    hidden_size = 768
    emb_size = 768
    max_layer = 8
    epoch = 80
    log_freq = 10
    use_fixed_gumbel = True

    processor = MnliProcessor(
        data_dir=data_dir,
        vocab_path=vocab_path,
        max_seq_len=max_seq_len,
        do_lower_case=do_lower_case,
        in_tokens=False)

    train_reader = processor.data_generator(
        batch_size=batch_size,
        phase='search_train',
        epoch=1,
        dev_count=1,
        shuffle=True)

    val_reader = processor.data_generator(
        batch_size=batch_size,
        phase='search_valid',
        epoch=1,
        dev_count=1,
        shuffle=True)

    if use_data_parallel:
        train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)
        valid_reader = fluid.contrib.reader.distributed_batch_reader(
            valid_reader)

    with fluid.dygraph.guard(place):
        model = AdaBERTClassifier(
            3,
            n_layer=max_layer,
            hidden_size=hidden_size,
            emb_size=emb_size,
            teacher_model=teacher_model_dir,
            data_dir=data_dir,
            use_fixed_gumbel=use_fixed_gumbel)

        if use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        device_num = fluid.dygraph.parallel.Env().nranks
        step_per_epoch = int(num_samples / (batch_size * device_num))
        learning_rate = fluid.dygraph.CosineDecay(2e-2, step_per_epoch, epoch)

        model_parameters = [
            p for p in model.parameters()
            if p.name not in [a.name for a in model.arch_parameters()]
        ]

        grad_clip = fluid.clip.GradientClipByGlobalNorm(5.0)
        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(3e-4),
            grad_clip=grad_clip,
            parameter_list=model_parameters)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=1024,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=1024,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(val_reader, places=place)

        architect = Architect(model, learning_rate, 3e-4, place, False)

        for epoch_id in range(epoch):
            train_one_epoch(model, architect, train_loader, valid_loader,
                            optimizer, epoch_id, use_data_parallel, log_freq)
            valid_one_epoch(model, valid_loader, epoch_id, log_freq)
            print(model.student._encoder.alphas.numpy())
            print("=" * 100)


if __name__ == '__main__':
    main()
