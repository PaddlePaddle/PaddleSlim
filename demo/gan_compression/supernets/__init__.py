import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    model_filename = "supernets." + model_name + "_supernet"
    modellib = importlib.import_module(model_filename)
    target_model_name = model_name.replace("_", "") + "supernet"
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


def create_supernet(cfg):
    supernet = find_model_using_name(cfg.supernet)
    return supernet(cfg)
