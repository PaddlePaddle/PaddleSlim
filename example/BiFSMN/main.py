import os
import paddle
import argparse
import random
import warnings
try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
except:
    print('no resource')
from train_valid_test import train_epoch_distill, valid_epoch_distill, train_epoch, valid_epoch, create_lr_schedule, create_optimizer, get_model, create_dataloader


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--saveroot', help=
        'set root folder for log and checkpoint', type=str, default=
        '/mnt/disk1/jsh/2030/paddle/BiFSMN-main/paddle_project/checkpoint')
    parser.add_argument('--dataroot', help='set root folder for dataset',
        type=str, default='/mnt/disk1/datasets/speech/')
    parser.add_argument('--checkpoint', help=
        'choose a checkpoint to resume', type=str, default=None)
    parser.add_argument('--test', action='store_true', help=
        'test accuracy with input checkpoint')
    parser.add_argument('--n_mels', type=int, default=32, help=
        'mel feature size')
    parser.add_argument('--model', type=str, default='Dfsmn')
    parser.add_argument('--dfsmn_with_bn', action='store_true', help=
        'use BatchNorm for Dfsmn model')
    parser.add_argument('--num_layer', type=int, default=8, help=
        'num_layer for  Dfsmn model')
    parser.add_argument('--frondend_channels', type=int, default=16, help=
        'frondend_channels for  Dfsmn model')
    parser.add_argument('--frondend_kernel_size', type=int, default=5, help
        ='frondend_kernel_size for  Dfsmn model')
    parser.add_argument('--hidden_size', type=int, default=256, help=
        'hidden_size for  Dfsmn model')
    parser.add_argument('--backbone_memory_size', type=int, default=128,
        help='backbone_memory_size for  Dfsmn model')
    parser.add_argument('--left_kernel_size', type=int, default=2, help=
        'left_kernel_size for  Dfsmn model')
    parser.add_argument('--right_kernel_size', type=int, default=2, help=
        'right_kernel_size for  Dfsmn model')
    parser.add_argument('--epoch', type=int, default=300, help='total epochs')
    parser.add_argument('--batch-size', type=int, default=768, help='batch size'
        )
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate'
        )
    parser.add_argument('--lr-scheduler', choices=['plateau', 'step',
        'cosin'], default='cosin', help='method to adjust learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help=
        'weight decay')
    parser.add_argument('--lr-scheduler-patience', type=int, default=5,
        help=
        'lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced'
        )
    parser.add_argument('--lr-scheduler-stepsize', type=int, default=5,
        help='lr scheduler step: number of epochs of learning rate decay.')
    parser.add_argument('--lr-scheduler-gamma', type=float, default=0.1,
        help='learning rate is multiplied by the gamma to decrease it')
    parser.add_argument('--optim', choices=['sgd', 'adam'], default='sgd',
        help='choices of optimization algorithms')
    parser.add_argument('--label_smoothing', type=float, default=0, help=
        'label_smoothing (float, optional): A float in [0.0, 1.0].')
    parser.add_argument('--mixup_alpha', type=float, default=0, help=
        'mixup alpha.')
    parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')
    parser.add_argument('--seed', default=None, type=int, help=
        'seed for initializing training. ')
    parser.add_argument('--num_classes', type=int, default=12, choices=[12,
        20, 35], help='num_classes for dataset')
    parser.add_argument('--version', default='speech_commands_v0.01',
        choices=['speech_commands_v0.01', 'speech_commands_v0.02'], type=
        str, help='dataset version')
    parser.add_argument('--thin_n', type=int, default=3, choices=[1, 2, 3, 
        4], help='ways for BiDfsmn_thinnable')
    parser.add_argument('--distill', action='store_true', help='disitll')
    parser.add_argument('--distill_alpha', type=float, default=0, help=
        'disitll alpha.')
    parser.add_argument('--teacher_model', choices=['Vgg19Bn',
        'Mobilenetv1', 'Mobilenetv2', 'BCResNet', 'Dfsmn', 'BiDfsmn',
        'BiDfsmn_thinnable', 'BiDfsmn_thinnable_pre'], type=str, default=
        'Dfsmn', help='teacher model')
    parser.add_argument('--teacher_model_checkpoint', type=str, help=
        'teacher pretrained model path: saveroot + teacher_model_checkpoint')
    parser.add_argument('--pretrained', action='store_true', help=
        'load the pre-trained teacher model')
    parser.add_argument('--select_pass', type=str, default='no', choices=[
        'no', 'low', 'high'], help='high-pass or low-pass for wavelet.')
    parser.add_argument('--J', type=int, default=1, help='scale of wavelet.')
    parser.add_argument('--method', type=str, default='no', help='bi method.')
    parser.add_argument('--pretrained_model_checkpoint', type=str, help=
        'pretrained model path: saveroot + pretrained_model_checkpoint')
    parsed_args = parser.parse_args()
    return parsed_args


def test_speech_commands(configs, gpu_id=None):

    # model = get_model(configs.model, in_channels=1, **(vars(configs)))
    model = get_model(configs.model,
                      in_channels=1,
                      teacher=True,
                      **(vars(configs)))
    
    print(model)
    nparams = sum(p.numel() for p in model.parameters() if p.trainable)

    names_params = {
        n: p.numel() * 1e-6 for n, p in model.named_parameters() if p.trainable
    }
    sorted_names_params = sorted(names_params.items(), key=lambda kv: kv[1], reverse=True)
    
    if configs.checkpoint is None:
        raise RuntimeError('Test mode must provide checkpoint')

    chpk = paddle.load(configs.checkpoint)

    model.set_state_dict(chpk)

    dataloader_test = create_dataloader('testing', configs, use_gpu=gpu_id is not None, version=configs.version)
    criterion = paddle.nn.CrossEntropyLoss(label_smoothing=configs.label_smoothing)
    
    if configs.distill:
        valid_loss, accuracy = valid_epoch_distill(model, criterion, dataloader_test, 
                                                   epoch=0, with_gpu=gpu_id is not None, 
                                                   log_iter=10, writer=None)
    else:
        valid_loss, accuracy = valid_epoch(model, criterion, dataloader_test, 
                                           epoch=0, with_gpu=gpu_id is not None, 
                                           log_iter=10, writer=None)

def train_speech_commands(configs, gpu_id=None):
    best_accuracy = 0
    best_accuracys = None
    epoch = 0

    use_gpu = paddle.is_compiled_with_cuda()
    if gpu_id is not None:
        paddle.set_device(f"gpu:{gpu_id}")

    model = get_model(configs.model, in_channels=1, **(vars(configs)))
    nparams = sum(p.numel() for p in model.parameters() if p.trainable)

    names_params = {
        n: p.numel() * 1e-6 for n, p in model.named_parameters() if p.trainable
    }
    sorted_names_params = sorted(names_params.items(), key=lambda kv: kv[1], reverse=True)

    teacher_model = None
    if configs.distill:
        teacher_model = get_model(configs.teacher_model, in_channels=1, **(vars(configs)))
        chpk = paddle.load(configs.teacher_model_checkpoint)
        teacher_model.set_state_dict(chpk)
    if configs.pretrained:
        chpk = paddle.load(configs.pretrained_model_checkpoint)
        model.set_state_dict(chpk)

    criterion = paddle.nn.CrossEntropyLoss(label_smoothing=configs.label_smoothing)

    optimizer = create_optimizer(configs, model)    
    if configs.checkpoint is not None:
        chpk = paddle.load(configs.checkpoint)
        best_accuracy = chpk['accuracy']
        epoch = chpk['epoch']
        model.set_state_dict(chpk['state_dict'])
        optimizer.set_state_dict(chpk['optimizer'])

    lr_scheduler = create_lr_schedule(configs, optimizer)

    dataloader_train = create_dataloader('training', configs, use_gpu, version=configs.version)
    dataloader_valid = create_dataloader('validation', configs, use_gpu, version=configs.version)

    # if gpu_id is not None:
    #     model = model.cuda(gpu_id)
    #     if teacher_model is not None:
    #         teacher_model = teacher_model.cuda(gpu_id)

    # writer = SummaryWriter(log_dir=os.path.join(configs.saveroot, 'Log'), flush_secs=10)

    # Train
    for cur_epoch in range(epoch, configs.epoch):

        if configs.distill:
            train_loss = train_epoch_distill(
                model, teacher_model, optimizer, criterion, dataloader_train, 
                epoch=cur_epoch, with_gpu=use_gpu, log_iter=10, writer=None,
                mixup_alpha=configs.mixup_alpha, distill_alpha=configs.distill_alpha,
                select_pass=configs.select_pass, J=configs.J, num_classes=configs.num_classes
            )
            valid_loss, accuracy = valid_epoch_distill(model, criterion, dataloader_valid,
                                                       cur_epoch, use_gpu, 10, None)
        else:
            train_loss = train_epoch(
                model, optimizer, criterion, dataloader_train, epoch=cur_epoch, 
                with_gpu=use_gpu, log_iter=10, writer=None, mixup_alpha=configs.mixup_alpha, 
                num_classes=configs.num_classes
            )
            valid_loss, accuracy = valid_epoch(model, criterion, dataloader_valid,
                                                cur_epoch, use_gpu, 10, None)

        if configs.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=valid_loss)
        else:
            lr_scheduler.step()

        if not isinstance(accuracy, list):
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                checkpoint = {
                    'epoch': cur_epoch,
                    'state_dict': model.state_dict(),
                    'accuracy': best_accuracy,
                    'optimizer': optimizer.state_dict(),
                }
                pth_name = f"{configs.model}_acc_{best_accuracy}_epoch_{cur_epoch}_lr_{configs.lr}_wd_{configs.weight_decay}_lrscheudle_{configs.lr_scheduler}_v{int(configs.version[-1:])}-{int(configs.num_classes)}"
                if configs.distill:
                    pth_name += f"_distill_{configs.distill_alpha}"
                if configs.select_pass != 'no':
                    pth_name += f"_{configs.select_pass}_J_{configs.J}"
                pth_name += '.pth'
                best_checkpoint_path = os.path.join(configs.saveroot, pth_name)
                paddle.save(checkpoint, best_checkpoint_path)
                configs.checkpoint = best_checkpoint_path
        else:
            if best_accuracys is None:
                best_accuracys = accuracy
            avg_accuracy = accuracy[0]
            if avg_accuracy > best_accuracy and min([x - y for x, y in zip(accuracy[:-1], accuracy[1:])]) > 0:
                best_accuracy = avg_accuracy
                best_accuracys = accuracy
                checkpoint = {
                    'epoch': cur_epoch,
                    'state_dict': model.state_dict(),
                    'accuracy': best_accuracy,
                    'optimizer': optimizer.state_dict(),
                }
                pth_name = f"{configs.model}_acc_{best_accuracy}_epoch_{cur_epoch}_lr_{configs.lr}_wd_{configs.weight_decay}_lrscheudle_{configs.lr_scheduler}_v{int(configs.version[-1:])}-{int(configs.num_classes)}"
                if configs.distill:
                    pth_name += f"_distill_{configs.distill_alpha}"
                if configs.select_pass != 'no':
                    pth_name += f"_{configs.select_pass}_J_{configs.J}"
                pth_name += '.pth'
                best_checkpoint_path = os.path.join(configs.saveroot, pth_name)
                paddle.save(checkpoint, best_checkpoint_path)
                configs.checkpoint = best_checkpoint_path

            print(f'train loss: {train_loss}, valid: best_accuracy: {best_accuracy}, cur_accuracy: {["%.4f%%" % (x * 100) for x in accuracy]}, best_accuracys: {["%.4f%%" % (x * 100) for x in best_accuracys]}, valid loss: {valid_loss}')

    test_speech_commands(configs, gpu_id)


if __name__ == '__main__':
    args = parse_args()
    if args.test:
        test_speech_commands(args, args.gpu)
    else:
        if args.seed is not None:
            random.seed(args.seed)
            paddle.seed(seed=args.seed)
            paddle.fluid.framework.cudnn_deterministic = True
            warnings.warn(
                'You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.'
                )
        if args.gpu is not None:
            warnings.warn(
                'You have chosen a specific GPU. This will completely disable data parallelism.'
                )
        os.makedirs(args.saveroot, exist_ok=True)
        os.makedirs(os.path.join(args.saveroot, 'Log'), exist_ok=True)
        train_speech_commands(args, gpu_id=args.gpu)
