import paddle
import numpy as np
from custom_setup_ops import quant_blockwise, dequant_blockwise

paddle.seed(2333)

a = paddle.uniform([2, 64])
print(a.detach().flatten())
a_nf4, abs_max = quant_blockwise(a, None, 64, "nf4")
#print("nf4 a: ", a_nf4.detach().flatten())
#print("abs max: ", abs_max)

a_dequant = dequant_blockwise(a_nf4, None, abs_max, 64,  "nf4")
print("dequant a: ", a_dequant.flatten())
