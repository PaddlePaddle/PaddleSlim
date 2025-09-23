import sys
import paddle
import paddle.nn as nn
import math

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