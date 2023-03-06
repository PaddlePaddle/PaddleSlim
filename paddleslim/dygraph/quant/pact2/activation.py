# Used for TI TIDL quant aware training, by HuGui

import numpy as np
import paddle
from paddle.nn import Layer, Hardtanh
from paddle.nn import functional as F
from functional import *
from range_utils import *


# Parametric Activation (PACT) with clip values being power of 2
# Supports learned mode, estimated mode or fixed range
class PAct2(Layer):
    # constants - default/init values
    PACT2_RANGE_SHRINK_DEFAULT = 0.01   # 0.01, means 0.01/100
    PACT2_RANGE_INIT = 8.0              # this is the starting range

    def __init__(self, inplace=False, signed=None, range_shrink_activations=PACT2_RANGE_SHRINK_DEFAULT, clip_range=None,
                 power2_activation_range=True, batch_quant=False, **kwargs):
        '''
        :param inplace: output is in same tensor as input if this is set
        :param signed: whether the cliping range is symmetirc aroudn 0 or not.
        :param range_shrink_activations: whether to shrink range (using histogram)
        :param clip_range: specify this to use a fixed usr provided range (range will not be estimated)
        :param power2_activation_range: convert range to nearest power of 2 values
        :param batch_quant: using current batch range during training. during validation, running average will be used
        :param kwargs:
        '''
        super().__init__()
        
        if (clip_range is not None) and (signed is not None):
            assert signed == (clip_range[0] < 0.0), 'the signed flag provided did not match the clip_range provided'

        self.inplace = inplace
        self.signed = signed if (clip_range is None) else (clip_range[0] < 0.0)
        self.range_shrink_activations = range_shrink_activations
        self.fixed_range = (clip_range is not None)
        self.eps = np.power(2.0, -16.0)
        self.power2_activation_range = power2_activation_range
        self.batch_quant = batch_quant

        clip_init = max(abs(np.array(clip_range))) if (clip_range is not None) else self.PACT2_RANGE_INIT
        clip_init2 = np.power(2.0, np.ceil(np.log2(clip_init)))

        default_clips = (-clip_init2, clip_init2) \
            if (self.signed == True or self.signed is None) else (0.0, clip_init2)
        self.register_buffer("clips_act", paddle.to_tensor(default_clips, dtype=np.float32))
        print(self.clips_act)
        input()

        # range_update_factor_min is the lower bound for exponential update factor.
        # using 0.0 will freeze the ranges, since the update_factor becomes too small after some time
        self.range_update_factor_min = 0.001
        self.register_buffer('num_batches_tracked', paddle.to_tensor(-1.0, dtype=paddle.float32))

    def forward(self, x, update_activation_range=True, enable=True):
        self.clips_batch = None
        if self.training and update_activation_range:
            self.num_batches_tracked += 1
            if not self.fixed_range:
                with paddle.no_grad():
                    self.clips_batch = self.update_clips_act(x.value())            
        if not enable:
            signed = self.clips_act[0] < 0.0 if (self.signed is None) else self.signed
            y = x if signed else F.relu(x)
        else:
            clips = self.get_clips_act()
            y = clamp_g(x, clips[0], clips[1], self.training, self.inplace, requires_grad=False)
        
        return y

    def __repr__(self):
        clips = self.get_clips_act()
        return '{}(inplace={}, signed={}, clips={})'.format(self.__class__.__name__, self.inplace, self.signed, clips)

    def freeze_range(self):
        self.fixed_range = True
    
    def update_clips_act(self, x):
        x_min, x_max = extrema_fast(src=x, range_shrink_percentile=self.range_shrink_activations)
        # exponential update factor
        update_factor = 1.0 / float(self.num_batches_tracked if self.num_batches_tracked else 1.0)
        update_factor = max(update_factor, self.range_update_factor_min)

        # exponential moving average update
        # paddle version
        self.clips_act[0] = self.clips_act[0] * (1.0 - update_factor) + (x_min * update_factor)
        self.clips_act[1] = self.clips_act[1] * (1.0 - update_factor) + (x_max * update_factor)
        # pytorch version
        # self.clips_act[0].data.mul_(1.0-update_factor).add_(x_min * update_factor)
        # self.clips_act[1].data.mul_(1.0-update_factor).add_(x_max * update_factor)

        clips_batch = paddle.to_tensor((x_min, x_max), place=self.clips_act.place)
        return clips_batch

    def get_clips_act(self):
        if (self.batch_quant and self.training and self.clips_batch is not None):
            clips = self.clips_batch
        else:
            clips = self.clips_act
        
        # find the clip values
        signed = clips[0] < 0.0 if (self.signed is None) else self.signed
        clip_max = paddle.max(paddle.abs(clips)) if signed else paddle.abs(clips[1])
        clip_max = paddle.clip(clip_max, min=self.eps)
        clip_max2 = ceil2_g(clip_max) if self.power2_activation_range else clip_max
        clip_min2 = (-clip_max2 if signed else clip_max2*0.0)
        return (clip_min2, clip_max2)


###############################################################
# return a function that creates PAct2 with the given fixed range
# remember: this function returns a type and not an instance
def get_fixed_pact2_type(inplace=False, signed=None, output_range=None):
        def FixedPAct2Type(inplace=inplace, signed=signed):
            assert output_range is not None, 'output_range must be specified for FixedPact2'
            clip_range = output_range #max(abs(np.array(output_range)))
            signed = True if ((output_range[0] < 0.0) or (signed is True)) else signed
            return PAct2(inplace=inplace, signed=signed, clip_range=clip_range)
        #
        return FixedPAct2Type 


###############################################################
# return a derivative of Hardtanh with the given fixed range
# remember: this function returns a type and not an instance
def get_fixed_hardtanh_type(*args, **kwargs):
        class FixedHardtanhType(Hardtanh):
            def __init__(self, *args_, **kwargs_):
                super().__init__(*args, **kwargs)
        #
        return FixedHardtanhType


###############################################################
class ReLU1(Hardtanh):
    def __init__(self, min_val=0., max_val=1., inplace=False):
        super().__init__(min=min_val, max=max_val, inplace=inplace)


###############################################################
# Always quantized activation function.
# Inserting this activation function is a simple way to ensure quantization happens at certain places.
class QAct(Layer):
    def __init__(self, inplace=False, signed=True, **kwargs):
        super().__init__()
        self.inplace = inplace
        self.signed = signed
    def forward(self, x):
        return x


###############################################################
# Never quantized activation function.
# Also if the next block is this, the previous block output is also not quantized.
# Inserting this activation function is a simple way to avoid quantization at certain places.
class NoQAct(Layer):
    def __init__(self, inplace=False, signed=True, **kwargs):
        super().__init__()
        self.inplace = inplace
        self.signed = signed
    def forward(self, x):
        return x


###############################################################
def freeze_quant_range(module):
    def _freeze_range_op(op):
        if isinstance(op, PAct2):
            op.freeze_range()
        #
    #
    module.apply(_freeze_range_op)
    # module.apply(torch.quantization.disable_observer)
