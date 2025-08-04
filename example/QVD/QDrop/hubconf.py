import paddle
from collections import OrderedDict
from QDrop.models.resnet import resnet18 as _resnet18
from QDrop.models.resnet import resnet50 as _resnet50
from QDrop.models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from QDrop.models.mnasnet import mnasnet as _mnasnet
from QDrop.models.regnet import regnetx_600m as _regnetx_600m
from QDrop.models.regnet import regnetx_3200m as _regnetx_3200m
dependencies = ['torch']
prefix = '/mnt/lustre/weixiuying'
model_path = {'resnet18': prefix + '/model_zoo/resnet18_imagenet.pth.tar',
    'resnet50': prefix + '/model_zoo/resnet50_imagenet.pth.tar', 'mbv2': 
    prefix + '/model_zoo/mobilenetv2.pth.tar', 'reg600m': prefix +
    '/model_zoo/regnet_600m.pth.tar', 'reg3200m': prefix +
    '/model_zoo/regnet_3200m.pth.tar', 'mnasnet': prefix +
    '/model_zoo/mnasnet.pth.tar', 'spring_resnet50': prefix +
    '/model_zoo/spring_resnet50.pth'}


def resnet18(pretrained=False, **kwargs):
    model = _resnet18(**kwargs)
    if pretrained:
        checkpoint = paddle.load(path=str(model_path['resnet18']))
        model.set_state_dict(state_dict=checkpoint)
    return model


def resnet50(pretrained=False, **kwargs):
    model = _resnet50(**kwargs)
    if pretrained:
        checkpoint = paddle.load(path=str(model_path['resnet50']))
        model.set_state_dict(state_dict=checkpoint)
    return model


def spring_resnet50(pretrained=False, **kwargs):
    model = _resnet50(**kwargs)
    if pretrained:
        checkpoint = paddle.load(path=str(model_path['spring_resnet50']))
        q = OrderedDict()
        for k, v in checkpoint.items():
            q[k[7:]] = v
        model.set_state_dict(state_dict=q)
    return model


def mobilenetv2(pretrained=False, **kwargs):
    model = _mobilenetv2(**kwargs)
    if pretrained:
        checkpoint = paddle.load(path=str(model_path['mbv2']))
        model.set_state_dict(state_dict=checkpoint['model'])
    return model


def regnetx_600m(pretrained=False, **kwargs):
    model = _regnetx_600m(**kwargs)
    if pretrained:
        checkpoint = paddle.load(path=str(model_path['reg600m']))
        model.set_state_dict(state_dict=checkpoint)
    return model


def regnetx_3200m(pretrained=False, **kwargs):
    model = _regnetx_3200m(**kwargs)
    if pretrained:
        checkpoint = paddle.load(path=str(model_path['reg3200m']))
        model.set_state_dict(state_dict=checkpoint)
    return model


def mnasnet(pretrained=False, **kwargs):
    model = _mnasnet(**kwargs)
    if pretrained:
        checkpoint = paddle.load(path=str(model_path['mnasnet']))
        model.set_state_dict(state_dict=checkpoint)
    return model
