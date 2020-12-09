import math
import paddle


def piecewise_decay(net, device_num, args):
    step = int(
        math.ceil(float(args.total_images) / (args.batch_size * device_num)))
    bd = [step * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=bd, values=lr, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        parameters=net.parameters(),
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def cosine_decay(net, device_num, args):
    step = int(
        math.ceil(float(args.total_images) / (args.batch_size * device_num)))
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=step * args.num_epochs, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        parameters=net.parameters(),
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay))
    return optimizer, learning_rate


def create_optimizer(net, device_num, args):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(net, device_num, args)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(net, device_num, args)
