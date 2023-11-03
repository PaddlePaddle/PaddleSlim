import paddle
import numpy as np
from custom_setup_ops import quant_blockwise, dequant_blockwise

paddle.seed(2333)

a = paddle.uniform([4, 64])
#print(a)
a_nf4, abs_max = quant_blockwise(a, None, 64, 4*64, "nf4")
#print("nf4 a: ", a_nf4)
#print("abs max: ", abs_max)

a_dequant = dequant_blockwise(a_nf4, None, abs_max, 64, 2*64, "nf4")
print("dequant a: ", a_dequant)
