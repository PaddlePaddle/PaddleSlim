
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from itertools import product
from quantizers.uniform import *


class MinMaxQuantConv2d(nn.Conv2D):
    """
    MinMax quantize weight and output
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 qmode = 'raw',
                 w_bit = 8,
                 a_bit = 8):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        #根据paddle 中vit系列模型的定义，只有PatchEmbedding 中有conv 
        self.qmode = qmode
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = False)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
    
    def forward_(self, x,w,b):
        w=w.astype(paddle.float32)
        b=b.astype(paddle.float32)
        return F.conv._conv_nd(
            x,
            w,
            bias=b,
            stride=self._stride,
            padding=self._updated_padding,
            padding_algorithm=self._padding_algorithm,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
            channel_dim=self._channel_dim,
            op_type=self._op_type,
            use_cudnn=self._use_cudnn,
            )
    def forward(self, x):
        if self.qmode == 'raw':
            out = self.forward_(x, self.weight, self.bias)
        elif self.qmode == "quant_forward":
            out=self.quant_forward(x)
        elif self.qmode == 'debug_only_quant_weight':
            out = self.debug_only_quant_weight(x)
        elif self.qmode == 'debug_only_quant_act':
            out = self.debug_only_quant_act(x)
        else:
            raise NotImplementedError
        return out
            
    def quant_weight_bias(self):
        w_sim = self.w_quantizer(self.weight)
        return w_sim, self.bias if self.bias is not None else None
    
    def quant_input(self,x):
        if self.a_quantizer.n_bits >= 8:
            return x
        return self.a_quantizer(x)
    
    def quant_forward(self,x):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = self.forward_(x_sim, w_sim, bias_sim)
        return out
    
    def debug_only_quant_weight(self, x):
        w_sim, bias_sim = self.quant_weight_bias()
        out = self.forward_(x, w_sim, bias_sim)
        return out
    
    def debug_only_quant_act(self, x):
        x_sim = self.quant_input(x)
        out = self.forward_(x_sim,self.weight, self.bias)
        return out
    
    
class PTQSLQuantConv2d(MinMaxQuantConv2d):
    """
    PTQSL on Conv2d
    weight: (oc,ic,kw,kh) -> (oc,ic*kw*kh) -> divide into sub-matrixs and quantize
    input: (B,ic,W,H), keep this shape

    Only support SL quantization on weights.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 qmode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, qmode, w_bit, a_bit)
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = True)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.metric = metric
        self.search_round = search_round
        self.eq_n = eq_n
        self.parallel_eq_n = eq_n
        scalea=paddle.zeros((self._out_channels, 1))
        self.w_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                    dtype=scalea.dtype,
                    default_initializer=paddle.nn.initializer.Assign(scalea))
        scaleb=paddle.zeros((1, 1, 1, 1))
        self.a_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                    dtype=scaleb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scaleb))
        # self.a_quantizer.register_buffer('scale', paddle.zeros((1, 1, 1, 1)))
    
    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        if metric == "mae":
            similarity = -paddle.abs(tensor_raw - tensor_sim)
        elif metric == "mse":
            similarity = -(tensor_raw - tensor_sim) ** 2
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity

    def quant_weight_bias(self):
        # self.weight_scale shape: (1, 1) or (oc, 1) 
        # self.weight       shape: (oc,ic,kw,kh)
        oc, ic, kw, kh = self.weight.data.shape
        w_sim = self.w_quantizer(self.weight.reshape([oc, ic * kw * kh])).reshape([oc, ic, kw, kh])
        return w_sim, self.bias if self.bias is not None else None

    
class PTQSLBatchingQuantConv2d(PTQSLQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 qmode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, qmode, w_bit, a_bit, metric, search_round, eq_n)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = self.raw_input.shape[0]
        if paddle.device.cuda.device_count() >= 1:
            props = paddle.device.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (2 * self.raw_input[:self.calib_batch_size].size + 
                 2 * self.raw_out[:self.calib_batch_size].size) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        self.parallel_eq_n = 1
        
    def _initialize_activation_scale(self):
        tmp_a_scales = []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed]
            a_scale_=(x_.abs().max() / (self.a_quantizer.n_levels - 0.5)).detach().reshape([1, 1])
            tmp_a_scales.append(a_scale_)
        tmp_a_scale = paddle.concat(tmp_a_scales, axis=1).amax(axis=1, keepdim=False).reshape([1, 1, 1, 1])
        self.a_quantizer.scale.data = paddle.assign(tmp_a_scale) # shape: (1, 1, 1, 1)
        self.a_quantizer.inited = True
        
    def _search_best_a_scale(self, input_scale_candidates):
        batch_similarities = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                B,ic,iw,ih = x.shape
                x_sim = x.unsqueeze(0) # shape: 1,B,ic,iw,ih
                x_sim = (x_sim / (cur_a_scale)).round_().clamp_(-self.a_quantizer.n_levels, self.a_quantizer.n_levels - 1) * cur_a_scale # shape: parallel_eq_n,B,ic,iw,ih
                x_sim = x_sim.reshape([-1,ic,iw,ih])
                # calculate similarity and store them
                out_sim = self.forward_(x_sim, w_sim, bias_sim) # shape: parallel_eq_n*B,oc,fw,fh
                out_sim = paddle.concat(paddle.chunk(out_sim.unsqueeze(0), chunks=p_ed-p_st, axis=1), axis=0) # shape: parallel_eq_n,B,oc,fw,fh
                out_sim = out_sim.transpose_(0, 1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = paddle.mean(similarity, axis=[2,3,4]) # shape: B,parallel_eq_n
                similarity = paddle.sum(similarity, axis=0, keepdim=True) # shape: 1,parallel_eq_n
                similarities.append(similarity)
            similarities = paddle.concat(similarities, axis=1) # shape: 1,eq_n
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=False) #shape: eq_n
        best_index = batch_similarities.argmax(axis=0).reshape([1,1,1,1,1])
        tmp_a_scale = paddle.gather(input_scale_candidates, axis=0, index=best_index.reshape([-1]))
        self.a_quantizer.scale.data = paddle.assign(tmp_a_scale.reshape([1,1,1,1]))
        
        
class AsymmetricallyBatchingQuantConv2d(PTQSLBatchingQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 qmode = 'raw',
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, qmode, w_bit, a_bit, metric, calib_batch_size, search_round, eq_n)

        del self.w_quantizer
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        scalea=paddle.zeros((self._out_channels, 1))
        zp=paddle.zeros((self._out_channels, 1))
        self.w_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                    dtype=scalea.dtype,
                    default_initializer=paddle.nn.initializer.Assign(scalea))
        self.w_quantizer.zero_point = paddle.create_parameter(shape=zp.shape,
                    dtype=zp.dtype,
                        default_initializer=paddle.nn.initializer.Assign(zp))
    def _search_best_w_scale(self, weight_scale_candidates, weight_zero_point_candidates):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed] # shape: (parallel_eq_n, 1, 1) or (parallel_eq_n, oc, 1)
                cur_w_zero_point = weight_zero_point_candidates[p_st:p_ed]
                # quantize weight and bias 
                oc,ic,kw,kh = self.weight.data.shape
                w_sim = self.weight.reshape([oc, -1]).unsqueeze(0) # shape: (1, oc, ic*kw*kh)
                w_quant = ((w_sim / cur_w_scale).round_() + cur_w_zero_point).clip(0, 2 * self.w_quantizer.n_levels - 1)
                w_sim = (w_quant - cur_w_zero_point).multiply(cur_w_scale) # shape: (parallel_eq_n,oc,ic*kw*kh)
                w_sim = w_sim.reshape([-1,ic,kw,kh]) # shape: parallel_eq_n*oc,ic,kw,kh
                bias_sim = self.bias.tile(p_ed-p_st) if self.bias is not None else None
                # quantize input
                x_sim = self.quant_input(x)
                # calculate similarity and store them
                out_sim = self.forward_(x_sim, w_sim, bias_sim) # shape: B,parallel_eq_n*oc,fw,fh
                out_sim = paddle.concat(paddle.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, axis=2), axis=1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = paddle.mean(similarity, [3,4]) # shape: B,parallel_eq_n,oc
                similarity = paddle.sum(similarity, axis=0, keepdim=True) # shape: (1,parallel_eq_n) or (1,parallel_eq_n,oc)
                similarities.append(similarity)
            similarities = paddle.concat(similarities, axis=1) # shape: (1,eq_n) or (1,eq_n,oc)
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=False) #shape: (eq_n) or (eq_n,oc)
        best_index = batch_similarities.argmax(axis=0).reshape([1, -1, 1]) # shape: (1,1,1) or (1,oc,1)
        tmp_w_scale=[]
        tmp_w_zero_point=[]
        for j in range(best_index.shape[1]):
            tmp_w_scale.append(weight_scale_candidates[best_index[0][j][0]][j][0])
            tmp_w_zero_point.append(weight_zero_point_candidates[best_index[0][j][0]][j][0])
        tmp_w_scale=paddle.to_tensor(tmp_w_scale)
        tmp_w_zero_point=paddle.to_tensor(tmp_w_zero_point)
        self.w_quantizer.scale.data = paddle.assign(tmp_w_scale.unsqueeze(-1))
        self.w_quantizer.zero_point.data = paddle.assign(tmp_w_zero_point.unsqueeze(-1))
        return best_index
    
    def calculate_percentile_weight_candidates(self, l=0.99, r=0.9999, k=0.05):
        pct = [l + (r - l) * (i / (self.eq_n - 1))**k for i in range(self.eq_n)] + [1.0]
        w_uppers_candidates = paddle.quantile(
            self.weight.reshape([self._out_channels, -1]), pct, axis=-1
        ).unsqueeze(-1) # shape: eq_n, out_channels, 1
        w_lowers_candidates = paddle.quantile(
            self.weight.reshape([self._out_channels, -1]), [1-pcti for pcti in pct], axis=-1
        ).unsqueeze(-1) # shapeL eq_n, out_channels, 1
        return w_uppers_candidates, w_lowers_candidates

    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        self._initialize_activation_scale()
        self.eq_alpha, self.eq_beta = 0.01, 1.2
        
        input_scale_candidates =  paddle.to_tensor(
            [self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]
        ).reshape([-1,1,1,1,1]) * self.a_quantizer.scale # shape: (eq_n,1,1,1,1)
        
        w_uppers_candidates, w_lowers_candidates = self.calculate_percentile_weight_candidates(l=0.99, r=0.9999, k=0.5)
        weight_scale_candidates = ((w_uppers_candidates - w_lowers_candidates) / (2 * self.w_quantizer.n_levels - 1)).contiguous()
        weight_zero_point_candidates = -(w_lowers_candidates / weight_scale_candidates).round().contiguous()
        w_best_index = self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
        self.w_quantizer.inited = True

        for e in range(self.search_round):
            for ee in range(2):
                if ee % 2 == 0:
                    w_uppers_candidates_=[]
                    for j in range(w_best_index.shape[1]):
                        w_uppers_candidates_.append(w_uppers_candidates[w_best_index[0][j][0]][j][0])
                    w_uppers_candidates_ = paddle.to_tensor(w_uppers_candidates_).reshape([1,-1,1])
                    w_lowers_candidates_ = w_lowers_candidates
                else:
                    w_lowers_candidates_=[]
                    for j in range(w_best_index.shape[1]):
                        w_lowers_candidates_.append(w_lowers_candidates[w_best_index[0][j][0]][j][0])
                    w_lowers_candidates_ = paddle.to_tensor(w_lowers_candidates_).reshape([1,-1,1])
                    w_uppers_candidates_ = w_uppers_candidates
                weight_scale_candidates = ((w_uppers_candidates_ - w_lowers_candidates_) / (2 * self.w_quantizer.n_levels - 1)).contiguous()
                weight_zero_point_candidates = -(w_lowers_candidates_ / weight_scale_candidates).round().contiguous()
                w_best_index = self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
            if self.a_quantizer.n_bits < 8:
                self._search_best_a_scale(input_scale_candidates)
            else:
                break
        self.calibrated = True
        del self.raw_input, self.raw_out
            