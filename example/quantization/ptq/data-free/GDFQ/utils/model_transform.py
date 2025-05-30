import paddle.nn as nn
import paddle
import numpy as np

__all__ = ["data_parallel", "model2list",
           "list2sequential", "model2state_dict"]


def data_parallel(model, ngpus, gpu0=0):
    return model


def model2list(model):
    """
    convert model to list type
    :param model: should be type of list or nn.DataParallel or nn.Sequential
    :return: no return params
    """
    if isinstance(model, nn.Sequential):
        model = list(model)
    return model


def list2sequential(model):
    if isinstance(model, list):
        model = nn.Sequential(*model)
    return model


def model2state_dict(file_path):
    model = paddle.load(file_path)
    if model['model'] is not None:
        model_state_dict = model['model'].state_dict()
        paddle.save(model_state_dict, file_path.replace(
            '.pth', 'state_dict.pth'))

    else:
        print((type(model)))
        print(model)
        print("skip")
