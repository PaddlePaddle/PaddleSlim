
import paddle
import numpy as np

from quant_train_module import QuantTrainPAct2
from activation import PAct2


if __name__ == "__main__":
    pact2 = QuantTrainPAct2(
        inplace=False, 
        signed=None, 
        clip_range=None, 
        bitwidth_weights=8, 
        bitwidth_activations=8,
        per_channel_q=False, 
        range_shrink_activations=0.01, 
        power2_weight_range=True,
        power2_activation_range=True
    )

    x = paddle.ones([4, 3, 32, 32])
    x_copy = x.clone()

    y = paddle.ones([4, 3, 32, 32])
    y = y * 127
    # print(x.set_value())
    
    x_copy.set_value(y)

    # paddle.assign(y, x_copy)
    # print(x_copy)
    x_inv = x.pow(-1.0)
    print(x.size)
    x = x * 127
    y = pact2(x)
    # print(y)
