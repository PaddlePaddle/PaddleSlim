import importlib
from .modules import *
from .base_model import BaseModel


def find_model_using_name(model_name):
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    target_model_name = model_name.replace('_', '')
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls,
                                                                    BaseModel):
            model = cls
    assert model is not None, "model {} is not right, please check it!".format(
        model_name)

    return model


def get_special_cfg(model):
    model_cls = find_model_using_name(model)
    return model_cls.add_special_cfgs


def create_model(cfg):
    model_cls = find_model_using_name(cfg.model)
    return model_cls(cfg)
