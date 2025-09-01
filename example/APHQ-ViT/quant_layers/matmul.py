import math
import paddle
from paddle import nn
from itertools import product
from quantizers.uniform import *
from datetime import datetime


class MinMaxQuantMatMul(nn.Layer):
    """Matrix Multiplication base class"""
    def __init__(self, A_bit=8, B_bit=8, qmode="raw"):
        super().__init__()
        self.qmode = qmode
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = False)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
    
    def forward(self, A, B):
        if self.qmode == 'raw':
            out = paddle.matmul(A,B)
        elif self.qmode == "quant_forward":
            A,B= self.quant_forward(A, B)
            out = paddle.matmul(A,B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input_A(self, x):
        return self.A_quantizer(x)
    
    def quant_input_B(self, x):
        return self.B_quantizer(x)
    

    def quant_forward(self, A, B):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        A = self.A_quantizer(A)
        B = self.B_quantizer(B)
        return A,B

    
    
class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    - Q @ K:
        - A's shape: B,H,S,C
        - B's shape: B,H,C,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,C
    """
    def __init__(self, A_bit=8, B_bit=8, qmode="raw", metric="mse", search_round=1, eq_n=100, 
                 head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, qmode)
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = head_channel_wise)
        self.metric = metric
        self.search_round = search_round
        self.eq_n = eq_n
        # the head dim is always dim-1
        self.head_channel_wise = head_channel_wise
        self.token_channel_wise = token_channel_wise
        self.num_heads = num_heads
        
        if not self.head_channel_wise:
            target_shape = [1, 1, 1, 1]
            scalea=paddle.zeros(target_shape)
            self.A_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                        dtype=scalea.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scalea))
            scaleb=paddle.zeros(target_shape)
            self.B_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                        dtype=scaleb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scaleb))
        else:
            target_shape = [1, self.num_heads, 1, 1]
            scalea=paddle.zeros(target_shape)
            self.A_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                        dtype=scalea.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scalea))
            scaleb=paddle.zeros(target_shape)
            self.B_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                        dtype=scaleb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scaleb))
    
    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        if metric == "mae":
            similarity = -paddle.abs(tensor_raw - tensor_sim)
        elif metric == "mse":
            similarity = -(tensor_raw - tensor_sim) ** 2
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity
        
    
class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, qmode="raw", metric="mse", calib_batch_size=32, 
                 search_round=1, eq_n=100, head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, qmode, metric, search_round, eq_n, head_channel_wise, token_channel_wise, num_heads)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = self.raw_input[0].shape[0]
        if paddle.device.cuda.device_count() >= 1:
            props = paddle.device.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (4 * self.raw_input[0][:self.calib_size].size+
                 4 * self.raw_input[1][:self.calib_size].size+
                 8 * self.raw_out[:self.calib_batch_size].size) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        self.parallel_eq_n = 1
        
        
class AsymmetricallyBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, qmode="raw", metric="mse", calib_batch_size=32, 
                 search_round=1, eq_n=100, head_channel_wise=True, token_channel_wise=False, num_heads=12):
        super().__init__(A_bit, B_bit, qmode, metric, calib_batch_size, search_round, 
                         eq_n, head_channel_wise, token_channel_wise, num_heads)
        del self.A_quantizer, self.B_quantizer
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = False, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = False, channel_wise = head_channel_wise)
        if not self.head_channel_wise:
            target_shape = [1, 1, 1, 1]
            scalea=paddle.zeros(target_shape)
            scaleb=paddle.zeros(target_shape)
            zpa=paddle.zeros(target_shape)
            zpb=paddle.zeros(target_shape)
            self.A_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                        dtype=scalea.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scalea))
            self.B_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                        dtype=scaleb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scaleb))
            self.A_quantizer.zero_point = paddle.create_parameter(shape=zpa.shape,
                        dtype=zpa.dtype,
                        default_initializer=paddle.nn.initializer.Assign(zpa))
            self.B_quantizer.zero_point = paddle.create_parameter(shape=zpb.shape,
                        dtype=zpb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(zpb))
        else:
            target_shape = [1, self.num_heads, 1, 1]
            scalea=paddle.zeros(target_shape)
            scaleb=paddle.zeros(target_shape)
            zpa=paddle.zeros(target_shape)
            zpb=paddle.zeros(target_shape)
            self.A_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                        dtype=scalea.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scalea))
            self.B_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                        dtype=scaleb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scaleb))
            self.A_quantizer.zero_point = paddle.create_parameter(shape=zpa.shape,
                        dtype=zpa.dtype,
                        default_initializer=paddle.nn.initializer.Assign(zpa))
            self.B_quantizer.zero_point = paddle.create_parameter(shape=zpb.shape,
                        dtype=zpb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(zpb))
    def _search_best_A_scale(self, A_scale_candidates, A_zero_point_candidates):
        paddle.device.cuda.empty_cache()
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed]
            B = self.raw_input[1][b_st:b_ed]
            B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,b,*,dim2,dim3
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0)
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize A
                cur_A_scale = A_scale_candidates[p_st:p_ed]
                cur_A_zero_point = A_zero_point_candidates[p_st:p_ed]
                A_sim = A.squeeze(0)
                A_quant = ((A_sim / cur_A_scale).round_() + cur_A_zero_point).clip(0, 2 * self.A_quantizer.n_levels - 1)
                A_sim = (A_quant - cur_A_zero_point).multiply(cur_A_scale) # shape: (parallel_eq_n,b,*,dim1,dim2)
                out_sim = paddle.matmul(A_sim.astype('float32'),B_sim.astype('float32')) # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = paddle.mean(similarity, axis=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = paddle.mean(similarity, axis=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(axis=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = paddle.concat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=1).sum(axis=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        best_index = paddle.argmax(batch_similarities, axis=0, keepdim=False).reshape([-1])
        tmp_A_scale = paddle.zeros_like(best_index,dtype='float32')
        tmp_A_zero_point = paddle.zeros_like(best_index,dtype='float32')
        #print(A_scale_candidates.shape,best_index.shape,"-------------------------------")
        for i in range(best_index.shape[0]):
            tmp_A_scale[i]=A_scale_candidates[best_index[i]][0][i][0][0]
            tmp_A_zero_point[i]=A_zero_point_candidates[best_index[i]][0][i][0][0]
        self.A_quantizer.scale.data = paddle.assign(tmp_A_scale.reshape(self.A_quantizer.scale.shape).astype('float32'))
        self.A_quantizer.zero_point.data = paddle.assign(tmp_A_zero_point.reshape(self.A_quantizer.zero_point.shape).astype('float32'))
        return best_index
        
    def _search_best_B_scale(self, B_scale_candidates, B_zero_point_candidates):
        paddle.device.cuda.empty_cache()
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed]
            B = self.raw_input[1][b_st:b_ed]
            A_sim = self.quant_input_A(A).unsqueeze(0) # shape: 1,b,*,dim1,dim2
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0)
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize B
                cur_B_scale = B_scale_candidates[p_st:p_ed]
                cur_B_zero_point = B_zero_point_candidates[p_st:p_ed]
                B_sim = B.squeeze(0)
                B_quant = ((B_sim / cur_B_scale).round_() + cur_B_zero_point).clip(0, 2 * self.B_quantizer.n_levels - 1)
                B_sim = (B_quant - cur_B_zero_point).multiply(cur_B_scale) # shape: (parallel_eq_n,b,*,dim2,dim3)
                out_sim = paddle.matmul(A_sim.astype('float32'),B_sim.astype('float32')) # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = paddle.mean(similarity, axis=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = paddle.mean(similarity, axis=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(axis=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = paddle.concat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = paddle.concat(batch_similarities, axis=1).sum(axis=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        best_index = paddle.argmax(batch_similarities, axis=0, keepdim=False).reshape([-1])
        tmp_B_scale = paddle.zeros_like(best_index,dtype='float32')
        tmp_B_zero_point = paddle.zeros_like(best_index,dtype='float32')
        for i in range(best_index.shape[0]):
            tmp_B_scale[i]=B_scale_candidates[best_index[i]][0][i][0][0]
            tmp_B_zero_point[i]=B_zero_point_candidates[best_index[i]][0][i][0][0]

        self.B_quantizer.scale.data = paddle.assign(tmp_B_scale.reshape(self.B_quantizer.scale.shape).astype('float32'))
        self.B_quantizer.zero_point.data = paddle.assign(tmp_B_zero_point.reshape(self.B_quantizer.zero_point.shape).astype('float32'))
        return best_index
    
    def calculate_percentile_candidates(self, x, l=0.99, r=0.99999):
        percentiles_uppers, percentiles_lowers = [], []
        pct = [l, r]
        tensor_too_large = True
        mini_batch_size = 1
        if self.head_channel_wise:
            x_ = x.transpose([1, 0,2,3]).contiguous() # shape: heads,b,*,dim1,dim2
            x_ = x_.reshape([x_.shape[0], mini_batch_size, -1]) 
        else:
            x_ = x.reshape([1, mini_batch_size, -1])
        
        while tensor_too_large:
            try:
                uppers_candidates = paddle.quantile(x_, pct, axis=-1).mean(axis=-1, keepdim=False) # shape: 2,(heads or 1)
                lowers_candidates = paddle.quantile(x_,  [1-pcti for pcti in pct], axis=-1).mean(axis=-1, keepdim=False) # shape: 2,(heads or 1)
                tensor_too_large = False
            except:
                mini_batch_size *= 2
                x_ = x_.reshape([x_.shape[0], mini_batch_size, -1]) if self.head_channel_wise else x_.reshape([1, mini_batch_size, -1])
        u_splits = paddle.linspace(0, 1, num=self.eq_n+1)[:, None, None, None, None] * (uppers_candidates[1] - uppers_candidates[0]).reshape([1, 1, -1, 1, 1])
        d_splits = paddle.linspace(0, 1, num=self.eq_n+1)[:, None, None, None, None] * (lowers_candidates[0] - lowers_candidates[1]).reshape([1, 1, -1, 1, 1])
        upper_candidates = uppers_candidates[0].reshape([1, 1, -1, 1, 1]) + u_splits
        lower_candidates = lowers_candidates[1].reshape([1, 1, -1, 1, 1]) + d_splits
        return upper_candidates , lower_candidates
        
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        A_uppers_candidates, A_lowers_candidates = self.calculate_percentile_candidates(self.raw_input[0], l=0.99, r=0.99999)
        B_uppers_candidates, B_lowers_candidates = self.calculate_percentile_candidates(self.raw_input[1], l=0.99, r=0.99999)
        A_scale_candidates = ((A_uppers_candidates - A_lowers_candidates) / (2 * self.A_quantizer.n_levels - 1)).contiguous()
        A_zero_point_candidates = -(A_lowers_candidates / A_scale_candidates).round().contiguous()
        B_scale_candidates = ((B_uppers_candidates - B_lowers_candidates) / (2 * self.B_quantizer.n_levels - 1)).contiguous()
        B_zero_point_candidates = -(B_lowers_candidates / B_scale_candidates).round().contiguous()
        self.A_quantizer.scale.data=paddle.assign(A_scale_candidates[-2])
        self.A_quantizer.zero_point.data=paddle.assign(A_zero_point_candidates[-2])
        self.B_quantizer.scale.data=paddle.assign(B_scale_candidates[-2])
        self.B_quantizer.zero_point.data=paddle.assign(B_zero_point_candidates[-2])
        self.A_quantizer.inited = True
        self.B_quantizer.inited = True

        A_best_index = self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
        B_best_index = self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        for e in range(self.search_round):
            if self.A_quantizer.n_bits < 32:
                for ee in range(2):
                    if ee % 2 == 0:
                        A_uppers_candidates_ = paddle.zeros_like(A_best_index,dtype='float32')
                        for i in range(A_best_index.shape[0]):
                            A_uppers_candidates_[i]=A_uppers_candidates[A_best_index[i]][0][i][0][0]
                        A_uppers_candidates_ = A_uppers_candidates_.reshape([1, 1, -1, 1, 1])
                        A_lowers_candidates_ = A_lowers_candidates
                    else:
                        A_uppers_candidates_ = A_uppers_candidates
                        A_lowers_candidates_ = paddle.zeros_like(A_best_index,dtype='float32')
                        for i in range(A_best_index.shape[0]):
                            A_lowers_candidates_[i]=A_lowers_candidates[A_best_index[i]][0][i][0][0]
                        A_lowers_candidates_ = A_lowers_candidates_.reshape([1, 1, -1, 1, 1])

                    A_scale_candidates = ((A_uppers_candidates_ - A_lowers_candidates_) / (2 * self.A_quantizer.n_levels - 1)).contiguous()
                    A_zero_point_candidates = -(A_lowers_candidates_ / A_scale_candidates).round().contiguous()
                    A_best_index = self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
            if self.B_quantizer.n_bits < 32:
                for ee in range(2):
                    if ee % 2 == 0:
                        B_uppers_candidates_ = paddle.zeros_like(B_best_index,dtype='float32')
                        for i in range(B_best_index.shape[0]):
                            B_uppers_candidates_[i]=B_uppers_candidates[B_best_index[i]][0][i][0][0]
                        B_uppers_candidates_ = B_uppers_candidates_.reshape([1, 1, -1, 1, 1])
                        B_lowers_candidates_ = B_lowers_candidates
                    else:
                        B_uppers_candidates_ = B_uppers_candidates
                        B_lowers_candidates_ = paddle.zeros_like(B_best_index,dtype='float32')
                        for i in range(B_best_index.shape[0]):
                            B_lowers_candidates_[i]=B_lowers_candidates[B_best_index[i]][0][i][0][0]
                        B_lowers_candidates_ = B_lowers_candidates_.reshape([1, 1, -1, 1, 1])
                    
                    B_scale_candidates = ((B_uppers_candidates_ - B_lowers_candidates_) / (2 * self.B_quantizer.n_levels - 1)).contiguous()
                    B_zero_point_candidates = -(B_lowers_candidates_ / B_scale_candidates).round().contiguous()
                    B_best_index = self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        
        if self.token_channel_wise:
            BA, HA, NA, MA = self.raw_input[0].shape
            BB, HB, NB, MB = self.raw_input[1].shape
            assert BA == BB and HA == HB and MA == NB
            A_token_wise_scale = self.A_quantizer.scale.expand(shape=[-1, -1, NA, -1])
            B_token_wise_scale = self.B_quantizer.scale.expand(shape=[-1, -1, -1, MB])
            del self.A_quantizer.scale, self.B_quantizer.scale

            scalea=A_token_wise_scale.clone()
            scaleb=B_token_wise_scale.clone()
            self.A_quantizer.scale = paddle.create_parameter(shape=scalea.shape,
                        dtype=scalea.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scalea))
            self.B_quantizer.scale = paddle.create_parameter(shape=scaleb.shape,
                        dtype=scaleb.dtype,
                        default_initializer=paddle.nn.initializer.Assign(scaleb))
        
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None
