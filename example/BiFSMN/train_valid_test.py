import paddle
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('.') 
from models.dfsmn import DfsmnModel
from models.bidfsmn import BiDfsmnModel, BiDfsmnModel_thinnable, DfsmnModel_pre
from speech_commands.dataset.speech_commands import SpeechCommandV1
from speech_commands.dataset.transform import ChangeAmplitude, FixAudioLength, ChangeSpeedAndPitchAudio, TimeshiftAudio

from utils.paddle_wavelets import DWTForward, DWTInverse


from paddle.vision.transforms import Compose, Normalize
import paddle.io as io
from paddle.io import DataLoader
import paddle.audio as audio
from paddle.audio.features import MelSpectrogram
from utils.AmplitudeToDB import AmplitudeToDB

import sys
import paddle
from typing import Tuple



def Count(module: nn.Layer, id=-1):
    id = 0 if id == -1 else id
    for name, child_module in module.named_sublayers():
        if isinstance(child_module, nn.LayerList):
            for child_child_module in child_module:
                id = Count(child_child_module, id)
        else:
            id = Count(child_module, id)
            if isinstance(child_module, nn.Linear):
                id += 1
            elif isinstance(child_module, nn.Conv1D):
                id += 1
            elif isinstance(child_module, nn.Conv2D):
                id += 1
    return id



def partial_mixup(input: paddle.Tensor, gamma: float, indices: paddle.Tensor
    ) ->paddle.Tensor:
    if input.shape[0] != indices.shape[0]:
        raise RuntimeError('Size mismatch!')
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: paddle.Tensor, target: paddle.Tensor, gamma: float,
    num_classes) ->Tuple[paddle.Tensor, paddle.Tensor]:
    target = paddle.nn.functional.one_hot(num_classes=num_classes, x=target
        ).astype('int64')
    indices = paddle.randperm(n=input.shape[0], dtype='int64')
    return partial_mixup(input, gamma, indices), partial_mixup(target,
        gamma, indices)


def naive_cross_entropy_loss(input: paddle.Tensor, target: paddle.Tensor
    ) ->paddle.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(axis=-1).mean()


def loss_term(A):
    a = paddle.abs(x=A)
    Q = a * a
    return Q


def total_loss(Q_s, Q_t):
    Q_s = loss_term(Q_s)
    Q_t = loss_term(Q_t)
    Q_s_norm = Q_s / paddle.linalg.norm(x=Q_s, p=2)
    Q_t_norm = Q_t / paddle.linalg.norm(x=Q_t, p=2)
    tmp = Q_s_norm - Q_t_norm
    loss = paddle.linalg.norm(x=tmp, p=2)
    return loss


def pass_filter(x, select_pass, J=1, wave='haar', mode='zero'):
    xfm = DWTForward(J=J, mode=mode, wave=wave)
    ifm = DWTInverse(mode=mode, wave=wave)
    
    # if 'gpu' in str(x.place):
    #     xfm, ifm = xfm.cuda(blocking=True), ifm.cuda(blocking=True)

    if len(tuple(x.shape)) == 3:
        yl, yh = xfm(x.unsqueeze(axis=1))
    elif len(tuple(x.shape)) == 4:
        yl, yh = xfm(x)
    else:
        assert False
    if select_pass == 'high':
        yl.zero_()
    y = ifm((yl, yh))
    if len(tuple(x.shape)) == 3:
        y = y.squeeze(axis=1)
    return y


def get_model2(model_type: str, in_channels=1, **kwargs):
    if model_type == 'Vgg19Bn':
        return Vgg19BN(in_channels=in_channels, **kwargs)
    elif model_type == 'Mobilenetv1':
        return MobileNetV1(in_channels=in_channels, **kwargs)
    elif model_type == 'Mobilenetv2':
        return MobileNetV2(in_channels=in_channels, **kwargs)
    elif model_type == 'BCResNet':
        return BCResNet(in_channels=in_channels, **kwargs)
    elif model_type == 'fsmn':
        return FSMN(in_channels=in_channels, **kwargs)
    elif model_type == 'Dfsmn':
        return DfsmnModel(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn':
        return BiDfsmnModel(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn_thinnable_pre':
        return DfsmnModel_pre(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn_thinnable':
        return BiDfsmnModel_thinnable(in_channels=in_channels, **kwargs)
    else:
        raise RuntimeError('unsupport model type: ', model_type)




def get_model(model_type: str, in_channels=1, method='no', **kwargs):
    if method == 'no':
        model = get_model2(model_type, in_channels, **kwargs)
        return model
    else:
        from binary_functions import Modify
        model = get_model2(model_type, in_channels, **kwargs)
        model.method = method
        cnt = Count(model)
        model, _ = Modify(model, method=method, id=0, first=31, last=cnt-30)
        return model



def create_dataloader(dataset_type, configs, use_gpu, version):
    # Define the transformations
    train_transform = Compose([
        # Assuming these are the correct paddles equivalents of your transformations
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        TimeshiftAudio(),
        FixAudioLength(),
        # MelSpectrogram(sr=16000, n_fft=2048, hop_length=512, n_mels=configs.n_mels),
        # AmplitudeToDB()
    ])
    
    valid_transform = Compose([
        FixAudioLength(),
        # MelSpectrogram(sr=16000, n_fft=2048, hop_length=512, n_mels=configs.n_mels),
        # AmplitudeToDB()
    ])
    
    # Load the datasets
    dataset_train = SpeechCommandV1(configs.dataroot,
                                    subset='training',
                                    download=False,
                                    transform=train_transform,
                                    num_classes=configs.num_classes,
                                    noise_ratio=0.3,
                                    noise_max_scale=0.3,
                                    cache_origin_data=False,
                                    version=version,
                                    config=configs)
    
    dataset_valid = SpeechCommandV1(configs.dataroot,
                                    subset='validation',
                                    download=False,
                                    transform=valid_transform,
                                    num_classes=configs.num_classes,
                                    cache_origin_data=True,
                                    version=version,
                                    config=configs
                                    )
    
    dataset_test = SpeechCommandV1(configs.dataroot,
                                   subset='testing',
                                   download=False,
                                   transform=valid_transform,
                                   num_classes=configs.num_classes,
                                   cache_origin_data=True,
                                   version=version,
                                   config=configs)

    # Create a dictionary of datasets
    dataset_dict = {
        'training': dataset_train,
        'validation': dataset_valid,
        'testing': dataset_test
    }
    
    # Create DataLoader
    dataloader = DataLoader(dataset_dict[dataset_type],
                            batch_size=configs.batch_size,
                            shuffle=(dataset_type == 'training'),
                            num_workers=16,
                            use_shared_memory=use_gpu,
                            persistent_workers=True)
    
    return dataloader

def create_lr_schedule(configs, optimizer):
    if configs.lr_scheduler == 'plateau':
        tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(patience=configs.
            lr_scheduler_patience, factor=configs.lr_scheduler_gamma,
            learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        lr_scheduler = tmp_lr
    elif configs.lr_scheduler == 'step':
        tmp_lr = paddle.optimizer.lr.StepDecay(step_size=configs.
            lr_scheduler_stepsize, gamma=configs.lr_scheduler_gamma,
            last_epoch=configs.epoch - 1, learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        lr_scheduler = tmp_lr
    elif configs.lr_scheduler == 'cosin':
        tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=configs.
            epoch, learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        lr_scheduler = tmp_lr
    else:
        raise RuntimeError('unsupported lr schedule type: ', configs.
            lr_scheduler)
    return lr_scheduler


def create_optimizer(configs, model):
    if configs.optim == 'sgd':
        optimizer = paddle.optimizer.SGD(
            learning_rate=configs.lr,
            # momentum=0.9,
            weight_decay=configs.weight_decay,
            parameters=model.parameters()
        )
    else:
        optimizer = paddle.optimizer.Adam(
            learning_rate=configs.lr,
            weight_decay=configs.weight_decay,
            parameters=model.parameters()
        )

    return optimizer


weights = [1, 0.5, 0.25]
loss_lim = 50.0
distillation_pred = paddle.nn.MSELoss()
pred = False


def train_epoch_distill(model: paddle.nn.Layer, teacher_model: paddle.nn.
    Layer, optimizer, criterion, data_loader: paddle.io.DataLoader, epoch,
    with_gpu, log_iter=10, writer=None, mixup_alpha=0, distill_alpha=0, select_pass='no', J=1,
    num_classes=None):
    """
    training one epoch
    """
    model.train()
    # if with_gpu:
    #     model = model.cuda(blocking=True)
    pbar = tqdm(data_loader, unit='audios', unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)
    running_loss = 0
    i = 0
    for inputs, target in pbar:
        if with_gpu:
            inputs = inputs.cuda(blocking=True)
            target = target.cuda(blocking=True)

        if 0 < mixup_alpha < 1:
            inputs, target = mixup.mixup(inputs, target, np.random.beta(
                mixup_alpha, mixup_alpha), num_classes)
        # print('inputs', inputs.shape)
        teacher_out, teacher_feature = teacher_model(inputs)
        if select_pass != 'no':
            teacher_feature = [(f1 / paddle.std(x=f1) + f2 / paddle.std(x=
                f2)) for f1, f2 in [(pass_filter(f, select_pass=select_pass,
                J=J), f) for f in teacher_feature]]
        loss = 0
        if model.__class__.__name__[-9:] != 'thinnable':
            out, feature = model(inputs)
            if 0 < mixup_alpha < 1:
                loss_one_hot = mixup.naive_cross_entropy_loss(out, target)
            else:
                loss_one_hot = criterion(out, target)
            if hasattr(model, 'method') and model.method == 'Laq':
                distr_loss1, distr_loss2 = model.laq_loss(inputs)
                distr_loss1 = distr_loss1.mean()
                distr_loss2 = distr_loss2.mean()
                if epoch < 100:
                    loss = loss + (distr_loss1 + distr_loss2)
            loss = loss + loss_one_hot
            if len(teacher_feature) % len(feature) == 0:
                loss_distill = None
                for k in range(len(feature)):
                    j = int(len(teacher_feature) / len(feature) * (k + 1) - 1)
                    if loss_distill == None:
                        loss_distill = total_loss(feature[k],
                            teacher_feature[j])
                    else:
                        loss_distill = loss_distill + total_loss(feature[k],
                            teacher_feature[j])
                loss = loss + loss_distill * distill_alpha
                if pred:
                    loss_pred = distillation_pred(out, teacher_out)
                    loss = loss + loss_pred * distill_alpha
            else:
                print('Distiilation Error: teacher {}, student {}!'.format(
                    len(teacher_feature), len(feature)))
        else:
            for op in range(model.thin_n):
                weight = weights[op]
                out, feature = model(inputs, op)
                if 0 < mixup_alpha < 1:
                    loss_one_hot = mixup.naive_cross_entropy_loss(out, target)
                else:
                    loss_one_hot = criterion(out, target)
                loss = loss + loss_one_hot * weight
                if len(teacher_feature) % len(feature) == 0:
                    loss_distill = None
                    for k in range(len(feature)):
                        j = int(len(teacher_feature) / len(feature) * (k + 
                            1) - 1)
                        if loss_distill is None:
                            loss_distill = total_loss(feature[k],
                                teacher_feature[j])
                        else:
                            loss_distill = loss_distill + total_loss(feature
                                [k], teacher_feature[j])
                    loss = loss + loss_distill * distill_alpha * weight
                    if pred:
                        loss_pred = distillation_pred(out, teacher_out)
                        loss = loss + loss_pred * distill_alpha * weight
                else:
                    print('Distiilation Error: teacher {}, student {}!'.
                        format(len(teacher_feature), len(feature)))
        optimizer.clear_gradients(set_to_zero=False)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % log_iter == 0:
            print(f"Train/iter_loss: {loss.item()}, iter: {i + epoch * epoch_size}")

        pbar.set_postfix({'loss': '%.05f' % loss.item()})
        i += 1

    running_loss /= i
    print(f"Train/epoch_loss: {running_loss}, epoch: {epoch}")
    return running_loss


def train_epoch(model: paddle.nn.Layer, optimizer, criterion, data_loader:
    paddle.io.DataLoader, epoch, with_gpu, log_iter=10, writer=None, mixup_alpha=0, num_classes=None):
    """
    training one epoch
    """
    model.train()
    # if with_gpu:
    #     model = model.cuda(blocking=True)
    pbar = tqdm(data_loader, unit='audios', unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)
    if model.__class__.__name__[-9:] != 'thinnable':
        running_loss = 0
        i = 0
        for feat, target in pbar:
            if with_gpu:
                feat = feat.cuda(blocking=True)
                target = target.cuda(blocking=True)
            if 0 < mixup_alpha < 1:
                feat, target = mixup.mixup(feat, target, np.random.beta(
                    mixup_alpha, mixup_alpha), num_classes)
            out = model(feat)
            if 0 < mixup_alpha < 1:
                loss = mixup.naive_cross_entropy_loss(out, target)
            else:
                loss = criterion(out, target)
            if hasattr(model, 'method') and model.method == 'Laq':
                distr_loss1, distr_loss2 = model.laq_loss(feat)
                distr_loss1 = distr_loss1.mean()
                distr_loss2 = distr_loss2.mean()
                if epoch < 100:
                    loss = loss + (distr_loss1 + distr_loss2)
            optimizer.clear_gradients(set_to_zero=False)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % log_iter == 0:
                print(f"Iteration {i + epoch * epoch_size} - Train/iter_loss: {loss.item()}")
            pbar.set_postfix({'loss': '%.05f' % loss.item()})
            i += 1
        running_loss /= i
        print(f"Epoch {epoch} - Train/epoch_loss: {running_loss}")
        return running_loss
    else:
        thin_n = model.thin_n
        running_loss = 0
        i = 0
        for inputs, target in pbar:
            if with_gpu:
                inputs = inputs.cuda(blocking=True)
                target = target.cuda(blocking=True)
            if 0 < mixup_alpha < 1:
                inputs, target = mixup.mixup(inputs, target, np.random.beta
                    (mixup_alpha, mixup_alpha), num_classes)
            loss = 0
            for op in range(thin_n):
                weight = weights[op]
                out = model(inputs, op)
                if 0 < mixup_alpha < 1:
                    loss += mixup.naive_cross_entropy_loss(out, target
                        ) * weight
                else:
                    loss += criterion(out, target) * weight
            optimizer.clear_gradients(set_to_zero=False)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % log_iter == 0:
                print(f"Iteration {i + epoch * epoch_size} - Train/iter_loss: {loss.item()}")
            pbar.set_postfix({'loss': '%.05f' % loss.item()})
            i += 1
        running_loss /= i
        print(f"Epoch {epoch} - Train/epoch_loss: {running_loss}")
        return running_loss


def valid_epoch_distill(model: paddle.nn.Layer, criterion, data_loader:
    paddle.io.DataLoader, epoch, with_gpu, log_iter=10, writer=None):
    """
    valid on dataset
    """
    model.eval()
    # if with_gpu:
    #     model = model.cuda(blocking=True)
    pbar = tqdm(data_loader, unit='audios', unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)
    if model.__class__.__name__[-9:] != 'thinnable':
        running_loss = 0
        running_acc = 0
        i = 0
        with paddle.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda(blocking=True)
                    target = target.cuda(blocking=True)
                out, feature = model(feat)
                loss = criterion(out, target)
                pred = paddle.argmax(out, axis=1, keepdim=True)
                acc = pred.equal(target.reshape(pred.shape)).sum(
                    ) / target.shape[0]
                running_loss += loss.item()
                running_acc += acc.item()
                if i % log_iter == 0:
                    print(f"Iteration {i + epoch * epoch_size} - Valid/iter_loss: {loss.item()}")
                pbar.set_postfix({'loss': '%.05f' % loss.item()})
                i += 1
        running_acc /= i
        running_loss /= i
        print(f"Epoch {epoch} - Valid/epoch_loss: {running_loss}")
        print(f"Epoch {epoch} - Valid/epoch_accuracy: {running_acc}")
        return running_loss, running_acc
    else:
        thin_n = model.thin_n
        running_loss = 0.0
        running_acc = [(0) for op in range(thin_n)]
        i = 0
        with paddle.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda(blocking=True)
                    target = target.cuda(blocking=True)
                for op in range(thin_n):
                    out, feature = model(feat, op)
                    loss = criterion(out, target)
                    pred = paddle.argmax(out, axis=1, keepdim=True)
                    acc = pred.equal(target.reshape(pred.shape)).sum(
                        ) / target.shape[0]
                    running_loss += loss.item()
                    running_acc[op] += acc.item()
                    if i % log_iter == 0:
                         print(f"Iter {i + epoch * epoch_size} - Valid/iter_loss[{[8, 4, 2, 1][op]}]: {loss.item()}")
                pbar.set_postfix({'loss': '%.05f' % loss.item()})
                i += 1
        running_acc = [(acc / i) for acc in running_acc]
        running_loss = running_loss / i

        print(f"Epoch {epoch} - Valid/epoch_loss: {running_loss}")
        for op in range(thin_n):
            print(f"Epoch {epoch} - Valid/epoch_accuracy_{[8, 4, 2, 1][op]}: {running_acc[op]}")

        return running_loss, running_acc


def valid_epoch(model: paddle.nn.Layer, criterion, data_loader: paddle.io.
    DataLoader, epoch, with_gpu, log_iter=10, writer=None):
    """
    valid on dataset
    """
    model.eval()
    # if with_gpu:
    #     model = model.cuda(blocking=True)
    pbar = tqdm(data_loader, unit='audios', unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)
    if model.__class__.__name__[-9:] != 'thinnable':
        running_loss = 0
        running_acc = 0
        i = 0
        with paddle.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda(blocking=True)
                    target = target.cuda(blocking=True)
                # print(F"FEAT: {feat.shape}, TARGET: {target.shape}")
                out = model(feat)
                loss = criterion(out, target)
                # print(f"target: {target.shape}, out: {out.shape}")
                # print('type(out): ', out.dtype)
                # pred = out.max(1, keepdim=True)
                pred = paddle.argmax(out, axis=1, keepdim=True)
                # print(f"pred: {pred}")
                # print('type(target): ', target.dtype)
                # print('type(pred): ', pred.dtype)
                acc = pred.equal(target.reshape(pred.shape)).sum() / target.shape[0]
                # acc = pred.equal(y=target.view_as(other=pred)).sum(
                #     ) / target.shape[0]
                running_loss += loss.item()
                running_acc += acc.item()

                if i % log_iter == 0:
                    print(f"Valid/iter_loss: {loss.item()}, Iteration: {i + epoch * epoch_size}")
                    
                pbar.set_postfix({'loss': '%.05f' % loss.item()})
                i += 1
        running_acc /= i
        running_loss /= i
        print(f"Epoch {epoch} - Valid/epoch_loss: {running_loss}, Valid/epoch_accuracy: {running_acc}")
        return running_loss, running_acc
    else:
        thin_n = model.thin_n
        running_loss = 0
        running_acc = [(0) for op in range(thin_n)]
        i = 0
        with paddle.no_grad():
            for feat, target in pbar:
                if with_gpu:
                    feat = feat.cuda(blocking=True)
                    target = target.cuda(blocking=True)
                for op in range(thin_n):
                    out = model(feat, op)
                    loss = criterion(out, target)
                    pred = paddle.argmax(out, axis=1, keepdim=True)
                    acc = pred.equal(target.reshape(pred.shape)).sum(
                        ) / target.shape[0]
                    running_loss += loss.item()
                    running_acc[op] += acc.item()
                    if i % log_iter == 0:
                         print(f"Iter {i + epoch * epoch_size} - Valid/iter_loss[{[8, 4, 2, 1][op]}]: {loss.item()}")
                pbar.set_postfix({'loss': '%.05f' % loss.item()})
                i += 1
        running_acc = [(acc / i) for acc in running_acc]
        running_loss = running_loss / i
        print(f"Epoch {epoch} - Valid/epoch_loss: {running_loss}")
        for op in range(thin_n):
            print(f"Epoch {epoch} - Valid/epoch_accuracy_{[8, 4, 2, 1][op]}: {running_acc[op]}")
        return running_loss, running_acc
