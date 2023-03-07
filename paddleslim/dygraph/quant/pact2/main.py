
import paddle
import numpy as np

from quant_train_module import QuantTrainPAct2
from activation import PAct2


if __name__ == "__main__":
    pact2 = QuantTrainPAct2(
        inplace=False, 
        signed=False, 
        clip_range=None, 
        bitwidth_weights=8, 
        bitwidth_activations=8,
        per_channel_q=False, 
        range_shrink_activations=0.01, 
        power2_weight_range=True,
        power2_activation_range=True
    )

    x = paddle.ones([4, 3, 16, 16])
    x = x * 3.5
    y, clips_act = pact2(x)
    print(clips_act)
