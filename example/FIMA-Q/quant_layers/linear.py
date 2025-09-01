import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from quantizers.uniform import *

'''
pytorch 与 paddle 的Linear运算差一个转置的区别：
pytorch: y=x*T(w)   w.shape=[out_f,in_f]=[n_v,crb_rows,in_f] (qkv层n_V=3,其它为1)
paddle:  y=x*w      w.shape=[in_f,out_f]=[in_f,n_v,crb_rows](qkv层n_V=3,其它为1)
'''
class MinMaxQuantLinear(nn.Linear):
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 bias: bool = True,
                 qmode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 weight_attr=None,
                 bias_attr=None):
        super().__init__(in_features, out_features, weight_attr=weight_attr,bias_attr=bias_attr)
        self.qmode = qmode
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = False)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        self.in_features=in_features
        self.out_features=out_features
    
    def forward_(self,x,weight,bias):
        x=x.astype(paddle.float32)
        weight=weight.astype(paddle.float32)
        #bias=bias.astype(paddle.float32)
        return F.linear(x, weight=weight, bias=bias, name=self.name)
    
    def forward(self, x):
        if self.qmode == 'raw':
            out = self.forward_(x=x, weight=self.weight, bias=self.bias)
        elif self.qmode == "quant_forward":
            out = self.quant_forward(x)
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

    def quant_input(self, x):
        return self.a_quantizer(x)
    
    def quant_forward(self,x):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = self.forward_(x=x_sim, weight=w_sim, bias=bias_sim)
        return out
    
    def debug_only_quant_weight(self, x):
        w_sim, bias_sim = self.quant_weight_bias()
        out = self.forward_(x=x, weight=w_sim, bias=bias_sim)
        return out
    
    def debug_only_quant_act(self, x):
        x_sim = self.quant_input(x)
        out = self.forward_(x=x_sim, weight=self.weight, bias=self.bias)
        return out
    

class PTQSLQuantLinear(MinMaxQuantLinear):
    """
    PTQSL on linear modules.
    """
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 qmode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 search_round = 1, 
                 eq_n = 100, 
                 n_V = 1, 
                 token_channel_wise=False,
                 weight_attr=None,
                 bias_attr=None):
        super().__init__(in_features, out_features, bias=bias, qmode=qmode, w_bit=w_bit, a_bit=a_bit,weight_attr=weight_attr,bias_attr=bias_attr)
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = True)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.metric = metric
        self.search_round = search_round
        self.eq_n = eq_n
        self.parallel_eq_n = eq_n
        self.n_V = n_V
        self.crb_rows = out_features // n_V
        self.token_channel_wise = token_channel_wise
        scalea=paddle.zeros((1,n_V, self.crb_rows))
        self.w_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                    dtype=scalea.dtype,
                    default_initializer=paddle.nn.initializer.Assign(scalea))
        scaleb=paddle.zeros((1))
        self.a_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                    dtype=scaleb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scaleb))

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        if metric == "mae":
            similarity = -paddle.abs(tensor_raw - tensor_sim)
        elif metric == "mse":
            similarity = -(tensor_raw - tensor_sim) ** 2
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity
    
    def quant_weight_bias(self):
        w_sim = self.w_quantizer(self.weight.reshape([self.in_features,self.n_V, self.crb_rows])).reshape([ self.in_features,self.out_features])
        return w_sim, self.bias if self.bias is not None else None


class PTQSLBatchingQuantLinear(PTQSLQuantLinear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 qmode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 n_V = 1, 
                 token_channel_wise=False,
                 weight_attr=None,
                 bias_attr=None,):
        super().__init__(in_features, out_features, bias=bias, qmode=qmode, w_bit=w_bit, a_bit=a_bit,
                         metric=metric, search_round=search_round, eq_n=eq_n, n_V=n_V, token_channel_wise=token_channel_wise,weight_attr=weight_attr,bias_attr=bias_attr)
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
        numel = (16 * self.raw_input[:self.calib_batch_size].size + 
                 16 * self.raw_out[:self.calib_batch_size].size) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        self.parallel_eq_n = 1
    
    def _initialize_weight_scale(self):
        self.w_quantizer.scale.data = paddle.assign(
            self.weight.reshape([ self.in_features,self.n_V, self.crb_rows]).abs().amax([0],keepdim=True) / 
            (self.w_quantizer.n_levels - 0.5)
        )
        self.w_quantizer.inited = True

    def _initialize_activation_scale(self):
        tmp_a_scales = []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st + self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed]
            a_scale_ = (x_.abs().max() / (self.a_quantizer.n_levels - 0.5)).detach().reshape([1, 1])
            tmp_a_scales.append(a_scale_)
        tmp_a_scale = paddle.concat(tmp_a_scales, axis=0).amax(axis=0, keepdim=False).reshape([-1])
        self.a_quantizer.scale.data = paddle.assign(tmp_a_scale)
        self.a_quantizer.inited = True

    def _search_best_w_scale(self, weight_scale_candidates):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]
            raw_out_expanded = self.raw_out[b_st:b_ed].unsqueeze(-2) # shape: b,*,1,out_features
            raw_out_expanded = raw_out_expanded.reshape([*raw_out_expanded.shape[:-1], self.n_V, -1]) # shape: b,*,1,n_V,crb_rows
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed]
                # quantize weight and bias 
                w_sim = self.weight.reshape([self.in_features,self.n_V, self.crb_rows]).unsqueeze(0) # shape: 1,in_features,n_V,crb_rows
                w_sim = (w_sim / cur_w_scale).round_().clip_(
                    -self.w_quantizer.n_levels, self.w_quantizer.n_levels - 1
                ).multiply(cur_w_scale) # shape: parallel_eq_n,in_features,n_V,crb_rows
                w_sim = w_sim.transpose([1,0,2,3]).reshape([self.in_features,-1]) # shape: in_features,parallel_eq_n*out_features
                bias_sim = self.bias.tile(p_ed - p_st) if self.bias is not None else None
                x_sim = self.quant_input(x)
                out_sim = self.forward_(x_sim, w_sim,bias=bias_sim) # shape: b,*,parallel_eq_n*out_features
                out_sim = out_sim.reshape([*out_sim.shape[:-1], p_ed-p_st, self.n_V, -1]) # shape: b,*,parallel_eq_n,n_V,crb_rows
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: b,*,parallel_eq_n,n_V,crb_rows
                if len(similarity.shape) > 4:
                    similarity = paddle.mean(similarity, axis=list(range(1,len(similarity.shape)-3))) # shape: b,parallel_eq_n,n_V,crb_rows
                similarity = similarity.sum(axis=0, keepdim=True) # shape: (1, parallel_eq_n, n_V) or (1, parallel_eq_n, n_V, crb_rows)
                similarities.append(similarity)
            similarities = paddle.concat(similarities, axis=1) # shape: (1, eq_n, n_V) or (1, eq_n, n_V, crb_rows)
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=False) # shape: (eq_n, n_V) or (eq_n, n_V, crb_rows)
        best_index = batch_similarities.argmax(axis=0).reshape([1, self.n_V, -1, 1]) # shape: (1,n_V,1,1) or (1,n_V,crb_rows,1)
        tmp_w_scale = paddle.gather(weight_scale_candidates, axis=0, index=reshaped_best_index) # shape: (1,n_V*crb_rows,1)
        self.w_quantizer.scale.data = paddle.assign(tmp_w_scale.squeeze(0))
        return best_index.squeeze(0) # shape: (n_V,crb_rows,1)

    def _search_best_a_scale(self, input_scale_candidates):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]
            raw_out_expanded = self.raw_out[b_st:b_ed].unsqueeze(-2) # shape: B,*,1,oc
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim = x.unsqueeze(-1) # shape: b,*,in_features,1
                x_sim = (x_sim / cur_a_scale).round_().clip_(
                    -self.a_quantizer.n_levels, self.a_quantizer.n_levels - 1
                ).multiply(cur_a_scale) # shape: B,*,in_features,parallel_eq_n
                x_sim = x_sim.permute(*list(range(len(x_sim.shape)-2)),-1,-2) # shape: B,*,parallel_eq_n,in_features
                # calculate similarity and store them
                out_sim = self.forward_(x_sim, w_sim, bias=bias_sim) # shape: B,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: B,*,parallel_eq_n,out_features
                similarity = paddle.mean(similarity, axis=-1) # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2:
                    similarity = paddle.mean(similarity, axis=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                similarity = paddle.sum(similarity, axis=0, keepdim=True) # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = paddle.concat(similarities, axis=1) # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=False) # shape: eq_n
        best_index = batch_similarities.argmax(axis=0, keepdim=True).reshape([1, -1])
        tmp_a_scale = paddle.gather(input_scale_candidates, axis=-1, index=best_index)
        self.a_quantizer.scale.data = paddle.assign(tmp_a_scale.squeeze(-1))
        return best_index.squeeze(0)

    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        self._initialize_weight_scale()
        self._initialize_activation_scale()

        # prepare weight scales and similarities
        self.eq_alpha, self.eq_beta = 0.01, 1.2
        weight_scale_candidates = paddle.to_tensor(
            [self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]
        ).reshape([-1,1,1,1]) * self.w_quantizer.scale.unsqueeze(0) # shape: eq_n,n_V,1,1
        input_scale_candidates =  paddle.to_tensor(
            [self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]
        ).reshape([1,-1]) * self.a_quantizer.scale.unsqueeze(-1) # shape: (1,eq_n) or (in_features,eq_n)
            
        for e in range(self.search_round):
            # search for best weight scale
            self._search_best_w_scale(weight_scale_candidates)
            # search for best input scale
            if self.a_quantizer.n_bits < 32:
                self._search_best_a_scale(input_scale_candidates)
            else:
                break

        self.calibrated = True
        del self.raw_input, self.raw_out
        return None
        
        
class AsymmetricallyBatchingQuantLinear(PTQSLBatchingQuantLinear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 qmode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 n_V = 1, 
                 token_channel_wise=False,
                 weight_attr=None,
                 bias_attr=None,
                 post_relu = False):
        super().__init__(in_features, out_features, bias=bias, qmode=qmode, w_bit=w_bit, a_bit=a_bit,
                         metric=metric, calib_batch_size=calib_batch_size, search_round=search_round, 
                         eq_n=eq_n, n_V=n_V, token_channel_wise=token_channel_wise,weight_attr=weight_attr,bias_attr=bias_attr)
        self.fix_zp_zero = post_relu
        
        del self.a_quantizer, self.w_quantizer
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = False, channel_wise = False)
        scalea=paddle.zeros((1))
        scaleb=paddle.zeros((1,n_V, self.crb_rows))
        zpa=paddle.zeros((1))
        zpb=paddle.zeros((1,n_V, self.crb_rows))
        self.a_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                    dtype=scalea.dtype,
                    default_initializer=paddle.nn.initializer.Assign(scalea))
        self.w_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                    dtype=scaleb.dtype,
                    default_initializer=paddle.nn.initializer.Assign(scaleb))
        self.a_quantizer.zero_point = paddle.create_parameter(shape=zpa.shape,
                    dtype=zpa.dtype,
                    default_initializer=paddle.nn.initializer.Assign(zpa))
        self.w_quantizer.zero_point = paddle.create_parameter(shape=zpb.shape,
                    dtype=zpb.dtype,
                    default_initializer=paddle.nn.initializer.Assign(zpb))

    def _initialize_weight_scale(self):
        self.w_quantizer.scale.data = paddle.assign(
            (self.weight.reshape([self.in_features,self.n_V, self.crb_rows]).amax([0],keepdim=True) - 
                self.weight.reshape([self.in_features,self.n_V, self.crb_rows]).amin([0],keepdim=True)) / 
            (2 * self.w_quantizer.n_levels - 1)
        )
        self.w_quantizer.zero_point.data = paddle.assign(
            -self.weight.reshape([self.in_features,self.n_V, self.crb_rows]).amin([0],keepdim=True) / self.w_quantizer.scale
        )
        self.w_quantizer.inited = True

    def _initialize_activation_scale(self):
        tmp_a_scales = []
        tmp_a_max, tmp_a_min = [], []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st + self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed]
            if self.a_quantizer.channel_wise:
                a_max = x_.abs().amax([i for i in range(x_.ndim-1)], keepdim=False).detach().reshape([1, -1])
                a_min = x_.abs().amin([i for i in range(x_.ndim-1)], keepdim=False).detach().reshape([1, -1])
            else:
                a_max = x_.abs().max().detach().reshape([1, 1])
                a_min = x_.abs().min().detach().reshape([1, 1])
            tmp_a_max.append(a_max)
            tmp_a_min.append(a_min)
        tmp_a_max = paddle.concat(tmp_a_max, axis=0).amax(axis=0, keepdim=False)
        tmp_a_min = paddle.concat(tmp_a_min, axis=0).amin(axis=0, keepdim=False)
        self.a_quantizer.scale.data = paddle.assign((tmp_a_max - tmp_a_min) / (2 * self.a_quantizer.n_levels - 1))
        self.a_quantizer.zero_point.data = paddle.assign(-tmp_a_min / self.a_quantizer.scale)
        self.a_quantizer.inited = True

    def _search_best_w_scale_self(self, weight_scale_candidates, weight_zero_point_candidates, topk=1):
        similarities = []
        raw_weight = self.weight.reshape([self.in_features,self.n_V, self.crb_rows]).unsqueeze(0) # shape: 1,in_features,n_V,crb_rows
        for p_st in range(0, self.eq_n, self.parallel_eq_n):
            p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
            cur_w_scale = weight_scale_candidates[p_st:p_ed]
            cur_w_zero_point = weight_zero_point_candidates[p_st:p_ed]
            # quantize weight and bias 
            w_quant = ((raw_weight / cur_w_scale).round_() + cur_w_zero_point).clip(0, 2 * self.w_quantizer.n_levels - 1)
            w_dequant = (w_quant - cur_w_zero_point) * cur_w_scale # shape: parallel_eq_n,in_features,n_V,crb_rows
            similarity = self._get_similarity(raw_weight, w_dequant, 'mse') # shape: parallel_eq_n,in_features,n_V,crb_rows
            similarity = paddle.mean(similarity, axis=1, keepdim=False) # shape: parallel_eq_n,n_V,crb_rows
            similarities.append(similarity)
        similarities = paddle.concat(similarities, axis=0) # shape: eq_n,n_V,crb_rows
        _, best_index = paddle.topk(similarities, k=topk, axis=0)
        best_index = best_index.reshape([topk, self.n_V, -1])
        
        if topk == 1:
            tmp_w_scale=paddle.zeros(best_index.shape)
            tmp_w_zero_point=paddle.zeros(best_index.shape)
            for i1 in range(best_index.shape[0]):
                for i2 in range(best_index.shape[1]):
                    for i3 in range(best_index.shape[2]):
                        tmp_w_scale[i1][i2][i3]=weight_scale_candidates[best_index[i1][i2][i3]][0][i2][i3]
                        tmp_w_zero_point[i1][i2][i3]=weight_zero_point_candidates[best_index[i1][i2][i3]][0][i2][i3]
            self.w_quantizer.scale.data = paddle.assign(paddle.to_tensor(tmp_w_scale))
            self.w_quantizer.zero_point.data = paddle.assign(paddle.to_tensor(tmp_w_zero_point))
            self.w_quantizer.inited = True
        return best_index.squeeze(0) # shape: (topk, n_V,crb_rows,1)

    def _search_best_a_scale_self(self, input_scale_candidates, input_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]
            raw_x = self.raw_input[b_st:b_ed].unsqueeze(-1) # shape: b,*,in_features,1
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                cur_a_zero_point = input_zero_point_candidates[:, p_st:p_ed]
                # quantize input
                x_sim = x.unsqueeze(-1) # shape: B,*,in_features,1
                x_quant = ((x_sim / cur_a_scale).round_() + cur_a_zero_point).clip_(0, 2 * self.a_quantizer.n_levels - 1) # shape: B,*,in_features,parallel_eq_n
                x_dequant = (x_quant - cur_a_zero_point) * cur_a_scale # shape: B,*,in_features,parallel_eq_n
                similarity = self._get_similarity(raw_x, x_dequant, 'mse') # shape: b,*,in_features,parallel_eq_n
                if len(similarity.shape) > 3:
                    similarity = paddle.mean(similarity, axis=list(range(1,len(similarity.shape)-2))) # shape: b, in_features, parallel_eq_n
                if not self.a_quantizer.channel_wise:
                    similarity = paddle.mean(similarity, axis=1, keepdim=True) # shape: b, 1, parallel_eq_n
                similarity = paddle.sum(similarity, axis=0, keepdim=True) # shape: 1, in_features, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = paddle.concat(similarities, axis=-1) # shape: 1, in_features, eq_n
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=False) # shape: in_features, eq_n
        _, best_index = paddle.topk(batch_similarities, k=topk, axis=-1) # shape: in_features, topk
        
        if topk == 1:
            tmp_a_scale = []
            tmp_a_zero_point = []
            for i in range(best_index.shape[0]):
                tmp_a_scale.append(input_scale_candidates[i][best_index[i][0]])
                tmp_a_zero_point.append(input_zero_point_candidates[i][best_index[i][0]])
            self.a_quantizer.scale.data = paddle.assign(paddle.to_tensor(tmp_a_scale))
            self.a_quantizer.zero_point.data = paddle.assign(paddle.to_tensor(tmp_a_zero_point))
            self.a_quantizer.inited = True
        return best_index
    
    def _search_best_w_scale(self, weight_scale_candidates, weight_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        paddle.device.cuda.empty_cache()
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]
            raw_out_expanded = self.raw_out[b_st:b_ed].unsqueeze(-2) # shape: b,*,1,out_features
            raw_out_expanded = raw_out_expanded.reshape([*raw_out_expanded.shape[:-1], self.n_V, -1]) # shape: b,*,1,n_V,crb_rows
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_scale = weight_scale_candidates[p_st:p_ed]
                cur_w_zero_point = weight_zero_point_candidates[p_st:p_ed]
                # quantize weight and bias 
                w_sim = self.weight.reshape([self.in_features,self.n_V, self.crb_rows]).unsqueeze(0) # shape: 1,in_features,n_V,crb_rows
                w_quant = ((w_sim / cur_w_scale).round_() + cur_w_zero_point).clip(0, 2 * self.w_quantizer.n_levels - 1)
                w_dequant = (w_quant - cur_w_zero_point) * cur_w_scale # shape: parallel_eq_n,in_features,n_V,crb_rows
                w_sim = w_dequant.transpose([1,0,2,3]).reshape([self.in_features,-1]) # shape: in_features,parallel_eq_n*out_features
                bias_sim = self.bias.tile(p_ed-p_st) if self.bias is not None else None
                x_sim = self.quant_input(x)
                out_sim = self.forward_(x_sim, w_sim,bias=bias_sim) # shape: B,*,parallel_eq_n*out_features
                out_sim = out_sim.reshape([*out_sim.shape[:-1], p_ed-p_st, self.n_V, -1]) # shape: b,*,parallel_eq_n,n_V,crb_rows
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: b,*,parallel_eq_n,n_V,crb_rows
                if len(similarity.shape) > 4:
                    similarity = paddle.mean(similarity, axis=list(range(1,len(similarity.shape)-3))) # shape: b, parallel_eq_n, n_V, crb_rows
                similarity = similarity.sum(axis=0, keepdim=True) # shape: (1, parallel_eq_n, n_V) or (1, parallel_eq_n, n_V, crb_rows)
                similarities.append(similarity)
            # store best weight scale of h into tmp_w_scale
            similarities = paddle.concat(similarities, axis=1) # shape: (1, eq_n, n_V) or (1, eq_n, n_V, crb_rows)
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=False) # shape: (eq_n, n_V) or (eq_n, n_V, crb_rows)
        _, best_index = paddle.topk(batch_similarities, k=topk, axis=0)
        best_index = best_index.reshape([topk, self.n_V, -1])
        if topk == 1:
            tmp_w_scale=paddle.zeros(best_index.shape)
            tmp_w_zero_point=paddle.zeros(best_index.shape)
            for i1 in range(best_index.shape[0]):
                for i2 in range(best_index.shape[1]):
                    for i3 in range(best_index.shape[2]):
                        tmp_w_scale[i1][i2][i3]=weight_scale_candidates[best_index[i1][i2][i3]][0][i2][i3]
                        tmp_w_zero_point[i1][i2][i3]=weight_zero_point_candidates[best_index[i1][i2][i3]][0][i2][i3]
            self.w_quantizer.scale.data = paddle.assign(paddle.to_tensor(tmp_w_scale))
            self.w_quantizer.zero_point.data = paddle.assign(paddle.to_tensor(tmp_w_zero_point))
        return best_index.squeeze(0) # shape: (topk, n_V,crb_rows,1)
    
    def _search_best_a_scale(self, input_scale_candidates, input_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed]
            raw_out_expanded = self.raw_out[b_st:b_ed].unsqueeze(-2) # shape: b,*,1,oc
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[:, p_st:p_ed]
                cur_a_zero_point = input_zero_point_candidates[:, p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim = x.unsqueeze(-1) # shape: B,*,in_features,1
                x_quant = ((x_sim / cur_a_scale).round_() + cur_a_zero_point).clip_(0, 2 * self.a_quantizer.n_levels - 1) # shape: B,*,in_features,parallel_eq_n
                x_dequant = (x_quant - cur_a_zero_point) * cur_a_scale # shape: B,*,in_features,parallel_eq_n
                if len(x_dequant.shape)==4:
                    x_sim = x_dequant.transpose([0,1,3,2]) # shape: B,*,parallel_eq_n,in_features
                else:
                    x_sim = x_dequant.transpose([0,2,1]) #for classifier
                # calculate similarity and store them
                out_sim = self.forward_(x_sim, w_sim,bias=bias_sim) # shape: b,*,parallel_eq_n,out_features
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: b,*,parallel_eq_n,out_features
                similarity = paddle.mean(similarity, axis=-1) # shape: B,*,parallel_eq_n
                if len(similarity.shape) > 2:
                    similarity = paddle.mean(similarity, axis=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                similarity = paddle.sum(similarity, axis=0, keepdim=True) # shape: 1, parallel_eq_n
                similarities.append(similarity)
            # store best input scale and store in tmp_a_scale
            similarities = paddle.concat(similarities, axis=1) # shape: 1, eq_n
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=0).sum(axis=0, keepdim=True) # shape: 1, eq_n
        _, best_index = paddle.topk(batch_similarities, k=topk, axis=-1) # shape: 1, topk
        if topk == 1:
            tmp_a_scale = input_scale_candidates[:,best_index[0][0]]
            tmp_a_zero_point = input_zero_point_candidates[:,best_index[0][0]]
            self.a_quantizer.scale.data = paddle.assign(paddle.to_tensor(tmp_a_scale))
            self.a_quantizer.zero_point.data = paddle.assign(paddle.to_tensor(tmp_a_zero_point))
        return best_index
        
    def calculate_percentile_weight_candidates(self, l=0.9, r=1.0):
        num_zp = min(16, self.w_quantizer.n_levels * 2)
        num_scale = int(self.eq_n / num_zp)
        pct =[l, r]
        w_uppers_candidates = paddle.quantile(
            self.weight.reshape([self.in_features,self.n_V, self.crb_rows]), pct, axis=0 #pytorch:self.weight.reshape([self.n_V, self.crb_rows, self.in_features]), pct, axis=-1
        ).unsqueeze(1) # shape: 2,1, n_V, crb_rows
        w_lowers_candidates = paddle.quantile(
            self.weight.reshape([self.in_features,self.n_V, self.crb_rows]), [1-pcti for pcti in pct], axis=0
        ).unsqueeze(1) # shapeL 2, 1,n_V, crb_rows
        delta_min = w_uppers_candidates[0:1] - w_lowers_candidates[0:1]
        delta_max = w_uppers_candidates[1:] - w_lowers_candidates[1:]
        splits = paddle.linspace(0, 1, num=num_scale)[:, None, None, None] * (delta_max - delta_min)
        weight_scale_candidates = (delta_min + splits).tile([num_zp, 1, 1, 1]) / (2 * self.w_quantizer.n_levels - 1)
        zp_min = int(self.w_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.w_quantizer.n_levels + num_zp / 2)
        zp_candidates = paddle.to_tensor(paddle.arange(zp_min, zp_max)).astype('float32') 
        weight_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[:, None, None, None]
        weight_zero_point_candidates = weight_zero_point_candidates.tile([self.in_features,1, self.n_V, self.crb_rows])
        return weight_scale_candidates, weight_zero_point_candidates
    
    def calculate_percentile_activation_candidates(self, l=0.9, r=1.0, fix_zp_zero=False):
        if fix_zp_zero:
            num_zp, num_scale = 1, self.eq_n
        else:
            num_zp = min(16, self.a_quantizer.n_levels * 2)
            num_scale = int(self.eq_n / num_zp)
        percentiles_uppers, percentiles_lowers = [], []
        pct = [l, r]
        x = self.raw_input
        tensor_too_large = True
        mini_batch_size = 1
        if self.a_quantizer.channel_wise:
            a_uppers_candidates = paddle.quantile(x.reshape([-1, x.shape[-1]]), pct, axis=0).transpose([1, 0]) # shape: in_features, 2
            a_lowers_candidates = paddle.quantile(x.reshape([-1, x.shape[-1]]), [1-pcti for pcti in pct], axis=0).transpose([1,0]) # shape: in_features, 2
        else:
            while tensor_too_large:
                try:
                    a_uppers_candidates = paddle.quantile(x.reshape([mini_batch_size, -1]), pct, axis=-1).mean(axis=-1).unsqueeze(0) # shape: 1, 2
                    a_lowers_candidates = paddle.quantile(x.reshape([mini_batch_size, -1]), [1-pcti for pcti in pct], axis=-1).mean(axis=-1).unsqueeze(0) # shape: 1, 2
                    tensor_too_large = False
                except:
                    mini_batch_size *= 2
        delta_min = a_uppers_candidates[:, 0:1] - a_lowers_candidates[:, 0:1]
        delta_max = a_uppers_candidates[:, 1:] - a_lowers_candidates[:, 1:]
        splits = paddle.linspace(0, 1, num=num_scale)[None, :] * (delta_max - delta_min)
        a_scale_candidates = ((delta_min + splits).tile([1, num_zp]) / (2 * self.a_quantizer.n_levels - 1)).clip(min=1e-4)
        a_scale_candidates = paddle.concat([a_scale_candidates, a_scale_candidates[..., -1:]], axis=-1)

        if fix_zp_zero:
            a_zero_point_candidates = paddle.zeros_like(a_scale_candidates)
        else:
            zp_min = int(self.a_quantizer.n_levels - num_zp / 2)
            zp_max = int(self.a_quantizer.n_levels + num_zp / 2)
            a=range(zp_min, zp_max)
            zp_candidates = paddle.to_tensor(paddle.arange(zp_min, zp_max)).astype('float32') 
            a_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[None, :]
            a_zero_point_candidates = a_zero_point_candidates.tile([a_scale_candidates.shape[0], 1])
            a_zero_point_candidates = paddle.concat([a_zero_point_candidates, a_zero_point_candidates[..., -1:]], axis=-1)
        return a_scale_candidates.astype('float32'), a_zero_point_candidates.astype('float32')
    
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()

        weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()
        a_scale_candidates, a_zero_point_candidates = self.calculate_percentile_activation_candidates()
        self._search_best_w_scale_self(weight_scale_candidates, weight_zero_point_candidates)
        self._search_best_a_scale_self(a_scale_candidates, a_zero_point_candidates)
        for e in range(self.search_round):
            paddle.device.cuda.empty_cache()
            self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
            self._search_best_a_scale(a_scale_candidates, a_zero_point_candidates)
        
        if (self.token_channel_wise and len(self.raw_input.shape) == 3):
            B, N, C = self.raw_input.shape
            token_wise_scale = self.a_quantizer.scale.expand([1, N, 1])
            del self.a_quantizer.scale
            scalea=token_wise_scale.clone()
            self.a_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                        dtype=scalea.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scalea))
        
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None
    

class AsymmetricallyChannelWiseBatchingQuantLinear(AsymmetricallyBatchingQuantLinear):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 qmode = "raw",
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 calib_batch_size = None,
                 search_round = 1, 
                 eq_n = 100, 
                 n_V=1,
                 token_channel_wise=False,
                 weight_attr=None,
                 bias_attr=None,
                 post_relu = False):
        super().__init__(in_features, out_features, bias=bias, qmode=qmode, w_bit=w_bit, a_bit=a_bit,
                         metric=metric, calib_batch_size=calib_batch_size, search_round=search_round, 
                         eq_n=eq_n, n_V=n_V, token_channel_wise=token_channel_wise,weight_attr=weight_attr,bias_attr=bias_attr, post_relu=post_relu)
        del self.a_quantizer
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = False, channel_wise = True)
        scale=paddle.zeros((in_features))
        zp=paddle.zeros((in_features))
        self.a_quantizer.scale = paddle.create_parameter(shape=scale.shape,
                        dtype=scale.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scale))
        self.a_quantizer.zero_point = paddle.create_parameter(shape=zp.shape,
                        dtype=zp.dtype,
                        default_initializer=paddle.nn.initializer.Assign(zp))
        self._prev_layer = None
    
    def __setattr__(self, name, value):
        if name == "prev_layer":
            self.__dict__['_prev_layer'] = value
        else:
            super().__setattr__(name, value)

    @property
    def prev_layer(self):
        return self._prev_layer

    @prev_layer.setter
    def prev_layer(self, layer):
        self._prev_layer = layer
    
    def hyperparameter_searching(self):
        assert self.a_quantizer.channel_wise and self.w_quantizer.channel_wise
        self._initialize_calib_parameters()
        a_scale_candidates, a_zero_point_candidates = self.calculate_percentile_activation_candidates(fix_zp_zero=self.fix_zp_zero)
        self._search_best_a_scale_self(a_scale_candidates, a_zero_point_candidates)
        self.calibrated = True
        
    def reparam_step1(self):
        self.calibrated = False
        channel_min = -self.a_quantizer.zero_point * self.a_quantizer.scale
        target_channel_scale = paddle.mean(self.a_quantizer.scale).reshape([1])
        target_channel_zero_point = paddle.mean(self.a_quantizer.zero_point).round().reshape([1])
        target_channel_min = -target_channel_zero_point * target_channel_scale
        r = (self.a_quantizer.scale / target_channel_scale)
        b = channel_min / r - target_channel_min
        if self.fix_zp_zero:
            self.prev_layer.weight.data = self.prev_layer.weight.data / r.reshape([-1, 1])
        else:
            self.prev_layer.weight.data = self.prev_layer.weight.data / r
        self.prev_layer.bias.data = self.prev_layer.bias.data / r.reshape([-1]) - b
        self.weight.data = self.weight.data * r.reshape([-1,1]) #.reshape([1, -1]
        
        if self.bias is not None:
            self.bias.data = self.bias.data + paddle.matmul(self.weight.data.transpose([1,0]), b.reshape([-1, 1])).reshape([-1])
        else:
            bias=paddle.zeros(self.out_features)
            self.bias = paddle.create_parameter(shape=bias.shape,
                        dtype=bias.dtype,
                        default_initializer=paddle.nn.initializer.Assign(bias))
            self.bias.data = paddle.matmul(self.weight.data.transpose([1,0]), b.reshape([-1, 1])).reshape([-1])
        return r, b, target_channel_scale, target_channel_zero_point
        
    def reparam(self):
        r, b, target_channel_scale, target_channel_zero_point = self.reparam_step1()
        self.raw_input = (self.raw_input / r - b).cpu()
        del self.a_quantizer.scale, self.a_quantizer.zero_point
        self.a_quantizer.channel_wise = False
        self.a_quantizer.scale = paddle.create_parameter(shape=target_channel_scale.shape,
                        dtype=target_channel_scale.dtype,
                        default_initializer=paddle.nn.initializer.Assign(target_channel_scale))
        self.a_quantizer.zero_point = paddle.create_parameter(shape=target_channel_zero_point.shape,
                        dtype=target_channel_zero_point.dtype,
                        default_initializer=paddle.nn.initializer.Assign(target_channel_zero_point))
        AsymmetricallyBatchingQuantLinear.hyperparameter_searching(self)
