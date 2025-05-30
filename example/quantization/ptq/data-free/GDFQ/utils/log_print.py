from termcolor import colored
import numpy as np
import datetime


__all__ = ["compute_remain_time", "print_result", "print_weight", "print_grad"]


single_train_time = 0
single_test_time = 0
single_train_iters = 0
single_test_iters = 0


def compute_remain_time(epoch, nEpochs, count, iters, data_time, iter_time, mode="Train"):
    global single_train_time, single_test_time
    global single_train_iters, single_test_iters

    # compute cost time
    if mode == "Train":
        single_train_time = single_train_time * \
                            0.95 + 0.05 * (data_time + iter_time)
        # single_train_time = data_time + iter_time
        single_train_iters = iters
        train_left_iter = single_train_iters - count + \
                          (nEpochs - epoch - 1) * single_train_iters
        # print "train_left_iters", train_left_iter
        test_left_iter = (nEpochs - epoch) * single_test_iters
    else:
        single_test_time = single_test_time * \
                           0.95 + 0.05 * (data_time + iter_time)
        # single_test_time = data_time+iter_time
        single_test_iters = iters
        train_left_iter = (nEpochs - epoch - 1) * single_train_iters
        test_left_iter = single_test_iters - count + \
                         (nEpochs - epoch - 1) * single_test_iters

    left_time = single_train_time * train_left_iter + \
                single_test_time * test_left_iter
    total_time = (single_train_time * single_train_iters +
                  single_test_time * single_test_iters) * nEpochs
    time_str = "TTime: {}, RTime: {}".format(datetime.timedelta(seconds=total_time),
                                                           datetime.timedelta(seconds=left_time))
    return time_str, total_time, left_time


def print_result(epoch, nEpochs, count, iters, lr, data_time, iter_time, error, loss, top5error=None,
                 mode="Train", logger=None):
    log_str = ">>> {}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}], LR: {:.6f}, DataTime: {:.4f}, IterTime: {:.4f}, ".format(
        mode, epoch + 1, nEpochs, count, iters, lr, data_time, iter_time)
    if isinstance(error, list) or isinstance(error, np.ndarray):
        for i in range(len(error)):
            log_str += "Error_{:d}: {:.4f}, Loss_{:d}: {:.4f}, ".format(i, error[i], i, loss[i])
    else:
        log_str += "Error: {:.4f}, Loss: {:.4f}, ".format(error, loss)

    if top5error is not None:
        if isinstance(top5error, list) or isinstance(top5error, np.ndarray):
            for i in range(len(top5error)):
                log_str += " Top5_Error_{:d}: {:.4f}, ".format(i, top5error[i])
        else:
            log_str += " Top5_Error: {:.4f}, ".format(top5error)

    time_str, total_time, left_time = compute_remain_time(epoch, nEpochs, count, iters, data_time, iter_time, mode)

    logger.info(log_str + time_str)

    return total_time, left_time


def print_weight(layers, logger):
    if isinstance(layers, MD.qConv2d):
        logger.info(layers.weight)
    elif isinstance(layers, MD.qLinear):
        logger.info(layers.weight)
        logger.info(layers.weight_mask)
    logger.info("------------------------------------")


def print_grad(m, logger):
    if isinstance(m, MD.qLinear):
        logger.info(m.weight.data)
