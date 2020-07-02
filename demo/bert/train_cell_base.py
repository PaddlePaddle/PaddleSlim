import numpy as np
from itertools import izip
import paddle.fluid as fluid
from paddleslim.teachers.bert import BERTClassifier
from paddleslim.teachers.bert.reader.cls import *
from paddleslim.nas.darts.search_space import AdaBERTClassifier
from paddleslim.nas.darts.architect_for_bert import Architect
from visualdl import LogWriter
import logging
import os
from paddleslim.common import AvgrageMeter, get_logger
logger = get_logger(__name__, level=logging.INFO)


def model_loss(model, a, b, labels):
    labels.stop_gradient = True

    logits = model.test(a, b)[-1]

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)
    return loss, accuracy


def count_parameters_in_MB(all_params):
    parameters_number = 0
    for param in all_params:
        if param.trainable:
            parameters_number += np.prod(param.shape)
    return parameters_number / 1e6


def train_one_epoch(model,
                    architect,
                    train_loader,
                    valid_loader,
                    optimizer,
                    epoch,
                    use_data_parallel,
                    log_freq,
                    scaler=None):
    ce_losses = AvgrageMeter()
    losses = AvgrageMeter()
    kd_losses = AvgrageMeter()
    accs = AvgrageMeter()
    model.train()
    #    for train_data in train_loader():
    #        loss, acc, ce_loss, kd_loss = model(train_data)
    #        batch_size = train_data[0].shape[0]
    #        break

    step_id = 0
    for a, b, label, logit, loss in train_loader():
        #        architect.step(train_data, valid_data)
        batch_size = a.shape[0]
        loss, acc, ce_loss, kd_loss = model(a, b, label, logit, loss)

        if scaler:
            loss = scaler.scale(loss)

        if use_data_parallel:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()
        if scaler:
            scaler.minimize(optimizer, loss)
        else:
            optimizer.minimize(loss)
        model.clear_gradients()

        losses.update(loss.numpy(), batch_size)
        kd_losses.update(kd_loss.numpy(), batch_size)
        ce_losses.update(ce_loss.numpy(), batch_size)
        accs.update(acc.numpy(), batch_size)

        if step_id % log_freq == 0:
            logger.info(
                "Train epoch={} step={} lr={:.6f} loss={:.6f} acc={:.6f} ce_loss={:.6f} kd_loss={:.6f}".
                format(epoch, step_id,
                       optimizer.current_step_lr(), losses.avg[0], accs.avg[0],
                       ce_losses.avg[0], kd_losses.avg[0]))
        step_id += 1


#    with LogWriter(logdir="./log") as writer:
#        writer.add_scalar(tag="train_loss", step=epoch, value=losses.avg[0])
#        writer.add_scalar(tag="train_acc", step=epoch, value=accs.avg[0])
#        writer.add_scalar(tag="train_ce_loss", step=epoch, value=ce_losses.avg[0])
#        writer.add_scalar(tag="train_kd_loss", step=epoch, value=kd_losses.avg[0])
#        writer.add_scalar(tag="lr", step=epoch, value=optimizer.current_step_lr())


def valid_one_epoch(model, valid_loader, epoch, log_freq):
    ce_losses = AvgrageMeter()
    accs = AvgrageMeter()
    model.eval()

    step_id = 0
    for a, b, labels, _, _ in valid_loader():
        loss, acc = model_loss(model, a, b, labels)

        batch_size = a.shape[0]
        ce_losses.update(loss.numpy(), batch_size)
        accs.update(acc.numpy(), batch_size)
        step_id += 1

    logger.info("Valid epoch={} loss={:.6f} acc={:.6f};".format(
        epoch, ce_losses.avg[0], accs.avg[0]))


#    with LogWriter(logdir="./log") as writer:
#        writer.add_scalar(tag="valid_loss", step=epoch, value=losses.avg[0])
#        writer.add_scalar(tag="valid_acc", step=epoch, value=accs.avg[0])


def main():
    use_data_parallel = True
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
    batch_size = 256
    hidden_size = 768
    emb_size = 768
    max_layer = 4
    epoch = 80
    log_freq = 10
    use_fixed_gumbel = True

    processor = MnliProcessor(
        data_dir=data_dir,
        vocab_path=vocab_path,
        max_seq_len=max_seq_len,
        do_lower_case=do_lower_case,
        in_tokens=False)

    #    train_reader = processor.data_generator(
    #        batch_size=batch_size,
    #        phase='train',
    #        epoch=1,
    #        dev_count=1,
    #        shuffle=True)

    teacher = BERTClassifier(3)
    train_reader = teacher.cache_reader(
        batch_size=batch_size, data_file="./teacher.train_data")
    teacher = BERTClassifier(3)
    val_reader = teacher.cache_reader(
        batch_size=batch_size, data_file="./teacher.dev_data")

    #    val_reader = processor.data_generator(
    #        batch_size=batch_size,
    #        phase='dev',
    #        epoch=1,
    #        dev_count=1,
    #        shuffle=False)

    if use_data_parallel:
        train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)
        valid_reader = fluid.contrib.reader.distributed_batch_reader(
            val_reader)

    with fluid.dygraph.guard(place):
        model = AdaBERTClassifier(
            3,
            n_layer=max_layer,
            hidden_size=hidden_size,
            emb_size=emb_size,
            teacher_model=teacher_model_dir,
            data_dir=data_dir,
            use_fixed_gumbel=use_fixed_gumbel,
            #            use_fixed_gumbel=False,
            gumbel_alphas=[
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0
            ],
            gamma=0.8)

        ori_model = model
        model_parameters = [p for p in model.student.parameters()]

        if use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        device_num = fluid.dygraph.parallel.Env().nranks
        step_per_epoch = int(num_samples / (batch_size * device_num))
        learning_rate = fluid.dygraph.CosineDecay(2e-2 * 10, step_per_epoch,
                                                  epoch)

        grad_clip = fluid.clip.GradientClipByGlobalNorm(5.0)
        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(3e-2),
            grad_clip=grad_clip,
            parameter_list=model_parameters)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=256 * 4,
            use_double_buffer=True,
            iterable=True,
            use_multiprocess=False,
            return_list=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=1024,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(val_reader, places=place)

        #architect = Architect(model, learning_rate, 3e-4, place, False)
        architect = None

        for epoch_id in range(epoch):
            train_one_epoch(model, architect, train_loader, valid_loader,
                            optimizer, epoch_id, use_data_parallel, log_freq)
            valid_one_epoch(ori_model, valid_loader, epoch_id, log_freq)


#
#        
#            save_path = os.path.join("./outputs", "epoch" + "_" + str(epoch_id))
#            fluid.save_dygraph(
#                    model.state_dict(),
#                    save_path)
#            fluid.save_dygraph(
#                    optimizer.state_dict(),
#                    save_path)
#            print("Save model parameters and optimizer status at %s" % save_path)
#
#            print(model.student._encoder.alphas.numpy())
#            print("=" * 100)
#

if __name__ == '__main__':
    main()
