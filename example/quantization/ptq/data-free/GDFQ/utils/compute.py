import numpy as np
import math

import paddle

__all__ = ["compute_tencrop", "compute_singlecrop", "AverageMeter"]


def compute_tencrop(outputs, labels):
    output_size = outputs.shape
    outputs = outputs.reshape([output_size[0] / 10, 10, output_size[1]])
    outputs = outputs.sum(1).squeeze(1)
    # compute top1
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t()
    top1_count = pred.equal(labels.reshape([1, -1]).expand_as(pred)).reshape([-1]).cast('float').sum(0).numpy()
    top1_error = 100.0 - 100.0 * top1_count / labels.shape[0]
    top1_error = float(top1_error.cpu().numpy())

    # compute top5
    _, pred = outputs.topk(5, 1, True, True)
    pred = pred.t()
    top5_count = pred.equal(labels.reshape([1, -1]).expand_as(pred)).reshape([-1]).cast('float').sum(0).numpy()
    top5_error = 100.0 - 100.0 * top5_count / labels.shape[0]
    top5_error = float(top5_error.cpu().numpy())
    return top1_error, 0, top5_error


def compute_singlecrop(outputs, labels, loss, top5_flag=False, mean_flag=False):
    with paddle.no_grad():
        if isinstance(outputs, list):
            top1_loss = []
            top1_error = []
            top5_error = []
            for i in range(len(outputs)):
                top1_accuracy, top5_accuracy = accuracy(outputs[i], labels, topk=(1, 5))
                top1_error.append(100 - top1_accuracy)
                top5_error.append(100 - top5_accuracy)
                top1_loss.append(loss[i].item())
        else:
            top1_accuracy, top5_accuracy = accuracy(outputs, labels, topk=(1,5))
            top1_error = 100 - top1_accuracy
            top5_error = 100 - top5_accuracy
            top1_loss = loss.item()

        if top5_flag:
            return top1_error, top1_loss, top5_error
        else:
            return top1_error, top1_loss

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.equal(target.reshape([1, -1]).expand_as(pred)).cast('float32')

        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).cast('float').sum(0, keepdim=True)
            # import pdb; pdb.set_trace()
            res.append((correct_k*(100.0 / batch_size)).item())
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        reset all parameters
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        update parameters
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count