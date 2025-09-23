import sys
import paddle
import numpy as np
from itertools import product


class MinMaxQuantMatMul(paddle.nn.Layer):
    """Matrix Multiplication base class"""

    def __init__(self, A_bit=8, B_bit=8, mode='raw'):
        super().__init__()
        self.A_bit = A_bit
        self.B_bit = B_bit
        self.A_interval = None
        self.B_interval = None
        self.A_qmax = 2 ** (self.A_bit - 1)
        self.B_qmax = 2 ** (self.B_bit - 1)
        self.mode = mode
        self.raw_input = None
        self.raw_out = None
        self.scale_initialized = None

    def forward(self, A, B):
        if self.mode == 'raw':
            out = A @ B
        elif self.mode == 'quant_forward':
            out = self.quant_forward(A, B)
        elif self.mode == 'calibration_step1':
            out = self.calibration_step1(A, B)
        elif self.mode == 'calibration_step2':
            out = self.calibration_step2(A, B)
        else:
            raise NotImplementedError
        return out

    def quant_forward(self, A, B):
        assert self.calibrated is not None, f'You should run calibrate_forward before run quant_forward for {self}'
        A_sim = self.quant_input_A(A)
        B_sim = self.quant_input_B(B)
        out = A_sim @ B_sim
        return out

    def calibration_step1(self, A, B):
        self.raw_input = A.cpu().detach(), B.cpu().detach()
        out = A @ B
        self.raw_out = out.cpu().detach()
        return out


class PTQSLBatchingQuantMatMul(MinMaxQuantMatMul):

    def __init__(self, A_bit=8, B_bit=8, mode='raw', metric='L2_norm',
        search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
        n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1,
        init_layerwise=False):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n
        self.n_G_A = n_G_A
        self.n_V_A = n_V_A
        self.n_H_A = n_H_A
        self.n_G_B = n_G_B
        self.n_V_B = n_V_B
        self.n_H_B = n_H_B
        self.crb_groups_A = None
        self.crb_groups_B = None
        self.crb_rows_A = None
        self.crb_cols_A = None
        self.crb_rows_B = None
        self.crb_cols_B = None
        self.pad_groups_A = None
        self.pad_groups_B = None
        self.pad_rows_A = None
        self.pad_rows_B = None
        self.pad_cols_A = None
        self.pad_cols_B = None
        self.raw_grad = None
        self.init_layerwise = init_layerwise
        self.A_zerop = None
        self.B_zerop = None

    def quant_input_A(self, x):
        x = paddle.nn.functional.pad(x=x, pad=[0, self.pad_cols_A, 0, self.
            pad_rows_A, 0, self.pad_groups_A], pad_from_left_axis=False)
        x = x.reshape([-1, self.n_G_A, self.crb_groups_A, self.n_V_A, self.
            crb_rows_A, self.n_H_A, self.crb_cols_A])
        if self.A_zerop is None:
            x = (x / self.A_interval).round_().clip(min=-self.A_qmax, max=
                self.A_qmax - 1).multiply_(y=paddle.to_tensor(self.A_interval))
        else:
            x = ((x / self.A_interval).round_() + self.A_zerop).clip(min=-
                self.A_qmax, max=self.A_qmax - 1)
            x = (x - self.A_zerop).multiply_(y=paddle.to_tensor(self.
                A_interval))
        x = x.reshape([-1, self.n_G_A * self.crb_groups_A, self.n_V_A * self.
            crb_rows_A, self.n_H_A * self.crb_cols_A])
        x = x[:, :tuple(x.shape)[1] - self.pad_groups_A, :tuple(x.shape)[2] -
            self.pad_rows_A, :tuple(x.shape)[3] - self.pad_cols_A]
        return x

    def quant_input_B(self, x):
        x = paddle.nn.functional.pad(x=x, pad=[0, self.pad_cols_B, 0, self.
            pad_rows_B, 0, self.pad_groups_B], pad_from_left_axis=False)
        x = x.reshape([-1, self.n_G_B, self.crb_groups_B, self.n_V_B, self.
            crb_rows_B, self.n_H_B, self.crb_cols_B])
        if self.B_zerop is None:
            x = (x / self.B_interval).round_().clip(min=-self.B_qmax, max=
                self.B_qmax - 1).multiply_(y=paddle.to_tensor(self.B_interval))
        else:
            x = ((x / self.B_interval).round_() + self.B_zerop).clip(min=-
                self.B_qmax, max=self.B_qmax - 1)
            x = (x - self.B_zerop).multiply_(y=paddle.to_tensor(self.
                B_interval))
        x = x.reshape([-1, self.n_G_B * self.crb_groups_B, self.n_V_B * self.
            crb_rows_B, self.n_H_B * self.crb_cols_B])
        x = x[:, :tuple(x.shape)[1] - self.pad_groups_B, :tuple(x.shape)[2] -
            self.pad_rows_B, :tuple(x.shape)[3] - self.pad_cols_B]
        return x

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(tuple(self.raw_input[0].shape)[0])
        self.calib_batch_size = int(tuple(self.raw_input[0].shape)[0])
        while True:
            numel = (self.raw_input[0].size + self.raw_input[1].size + 2 *
                self.raw_out.size) / self.calib_size * self.calib_batch_size
            self.parallel_eq_n = int(3 * 1024 * 1024 * 1024 / 4 // numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break

    def _get_padding_parameters(self, A, B):
        """
        We adopt a head-wise quantization here
        """
        self.n_G_A = tuple(A.shape)[1]
        self.n_G_B = tuple(B.shape)[1]
        self.crb_groups_A = (tuple(A.shape)[1] + self.n_G_A - 1) // self.n_G_A
        self.crb_groups_B = (tuple(B.shape)[1] + self.n_G_B - 1) // self.n_G_B
        self.crb_rows_A = (tuple(A.shape)[2] + self.n_V_A - 1) // self.n_V_A
        self.crb_cols_A = (tuple(A.shape)[3] + self.n_H_A - 1) // self.n_H_A
        self.crb_rows_B = (tuple(B.shape)[2] + self.n_V_B - 1) // self.n_V_B
        self.crb_cols_B = (tuple(B.shape)[3] + self.n_H_B - 1) // self.n_H_B
        self.pad_groups_A = self.crb_groups_A * self.n_G_A - tuple(A.shape)[1]
        self.pad_rows_A = self.crb_rows_A * self.n_V_A - tuple(A.shape)[2]
        self.pad_cols_A = self.crb_cols_A * self.n_H_A - tuple(A.shape)[3]
        self.pad_groups_B = self.crb_groups_B * self.n_G_B - tuple(B.shape)[1]
        self.pad_rows_B = self.crb_rows_B * self.n_V_B - tuple(B.shape)[2]
        self.pad_cols_B = self.crb_cols_B * self.n_H_B - tuple(B.shape)[3]

    def _initialize_intervals(self):
        if self.scale_initialized != True:
            self._get_padding_parameters(self.raw_input[0], self.raw_input[1])
            tmp_A_intervals = []
            tmp_B_intervals = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A, B = self.raw_input[0][b_st:b_ed].cuda(blocking=True
                    ), self.raw_input[1][b_st:b_ed].cuda(blocking=True)
                if self.init_layerwise:
                    A_interval = (A.abs().max() / (self.A_qmax - 0.5)).detach(
                        ).reshape([1, 1, 1, 1, 1, 1, 1]).tile(repeat_times=[1,
                        self.n_G_A, 1, self.n_V_A, 1, self.n_H_A, 1])
                    B_interval = (B.abs().max() / (self.B_qmax - 0.5)).detach(
                        ).reshape([1, 1, 1, 1, 1, 1, 1]).tile(repeat_times=[1,
                        self.n_G_B, 1, self.n_V_B, 1, self.n_H_B, 1])
                else:
                    A_pad = paddle.nn.functional.pad(x=A, pad=[0, self.
                        pad_cols_A, 0, self.pad_rows_A, 0, self.
                        pad_groups_A], pad_from_left_axis=False).unsqueeze(axis
                        =0).reshape([1, -1, self.n_G_A, self.crb_groups_A, self
                        .n_V_A, self.crb_rows_A, self.n_H_A, self.crb_cols_A])
                    B_pad = paddle.nn.functional.pad(x=B, pad=[0, self.
                        pad_cols_B, 0, self.pad_rows_B, 0, self.
                        pad_groups_B], pad_from_left_axis=False).unsqueeze(axis
                        =0).reshape([1, -1, self.n_G_B, self.crb_groups_B, self
                        .n_V_B, self.crb_rows_B, self.n_H_B, self.crb_cols_B])
                    A_interval = (A_pad.abs().amax(axis=[0, 1, 3, 5, 7],
                        keepdim=True) / (self.A_qmax - 0.5)).detach().squeeze(
                        axis=0)
                    B_interval = (B_pad.abs().amax(axis=[0, 1, 3, 5, 7],
                        keepdim=True) / (self.B_qmax - 0.5)).detach().squeeze(
                        axis=0)
                tmp_A_intervals.append(A_interval)
                tmp_B_intervals.append(B_interval)
            self.A_interval = paddle.concat(x=tmp_A_intervals, axis=0).amax(
                axis=0, keepdim=True)
            self.B_interval = paddle.concat(x=tmp_B_intervals, axis=0).amax(
                axis=0, keepdim=True)
        self.scale_initialized = True

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, dim=-1,
        raw_grad=None):
        """
        tensor_raw: *, features, *
        tensor_sim: *, features, *
        similarity: *
        It's your job to calculate mean on non-feature * dims!

        Similarity without inherent feature structure is more welcome to parallelism.
        """
        if metric == 'info':
            feat = tensor_raw - tensor_sim
            if len(tuple(feat.shape)) == 5:
                similarity = paddle.zeros(shape=tuple(feat.shape)[:3])
                a1, a2, a3 = tuple(feat.shape)[:3]
                feat = feat.reshape(a1, a2, a3, -1)
                _mean = paddle.mean(x=paddle.abs(x=feat), axis=-1)
                _std = paddle.std(x=feat, axis=-1)
                thresh = _std * 2
                for b1 in range(a1):
                    for b2 in range(a2):
                        for b3 in range(a3):
                            _feat = feat[b1][b2][b3]
                            _feat[_feat < thresh[b1][b2][b3]] = 0
                            outliers = paddle.sum(x=paddle.abs(x=_feat))
                            similarity[b1][b2][b3] = -outliers * _mean[b1][b2][
                                b3] * _std[b1][b2][b3]
        elif metric == 'cosine':
            similarity = paddle.nn.functional.cosine_similarity(x1=
                tensor_raw, x2=tensor_sim, axis=dim)
        elif metric == 'pearson':
            similarity = paddle.nn.functional.cosine_similarity(x1=
                tensor_raw - paddle.mean(x=tensor_raw, axis=dim, keepdim=
                True), x2=tensor_sim - paddle.mean(x=tensor_sim, axis=dim,
                keepdim=True), axis=dim)
        else:
            if metric == 'L1_norm':
                similarity = -paddle.abs(x=tensor_raw - tensor_sim)
            elif metric == 'L2_norm':
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == 'linear_weighted_L2_norm':
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == 'square_weighted_L2_norm':
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == 'apq_hessian':
                assert raw_grad is not None, f'No raw_grad in PTQSLBatchingQuantMatMul!'
                raw_grad = raw_grad.reshape(tensor_raw.shape)
                feat = tensor_raw - tensor_sim
                diff = paddle.abs(x=feat)
                perc = np.percentile(diff.detach().cpu().numpy(), 10)
                feat[diff < perc] = 0
                similarity = -(raw_grad * feat) ** 2
                # raw_grad = paddle.reshape(raw_grad, shape=tensor_raw.shape)
                # similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f'metric {metric} not implemented!')
            similarity = paddle.mean(x=similarity, axis=dim)
        return similarity

    def _search_best_A_interval(self, A_interval_candidates=None):
        """
        Modularization of searching best interval
        """
        tmp_A_interval = self.A_interval.unsqueeze(axis=0)
        if self.A_zerop is None:
            tmp_A_zerop = paddle.zeros_like(x=tmp_A_interval)
        else:
            tmp_A_zerop = self.A_zerop.unsqueeze(axis=0)
        for v, h in product(range(self.n_V_A), range(self.n_H_A)):
            batch_similarities = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input[0][b_st:b_ed].cuda(blocking=True)
                A_pad = paddle.nn.functional.pad(x=A, pad=[0, self.
                    pad_cols_A, 0, self.pad_rows_A, 0, self.pad_groups_A],
                    pad_from_left_axis=False).unsqueeze(axis=0).reshape([1, -1,
                    self.n_G_A, self.crb_groups_A, self.n_V_A, self.
                    crb_rows_A, self.n_H_A, self.crb_cols_A])
                A_max = A_pad.amax(axis=[0, 1, 3, 5, 7], keepdim=True).detach()
                A_min = A_pad.amin(axis=[0, 1, 3, 5, 7], keepdim=True).detach()
                new_max = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 
                    1, 1, 1, 1]) * A_max
                new_min = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 
                    1, 1, 1, 1]) * A_min
                new_scale = (new_max - new_min) / float(self.A_qmax * 2 - 1)
                new_scale.clip_(min=1e-08)
                new_zero_point = -self.A_qmax - paddle.round(new_min /
                    new_scale)
                new_zero_point.clip_(min=-self.A_qmax, max=self.A_qmax - 1)
                A_zeropoint_candidates = new_zero_point.cuda(blocking=True)
                A_interval_candidates = new_scale.cuda(blocking=True)
                B = self.raw_input[1][b_st:b_ed].cuda(blocking=True)
                B_sim = self.quant_input_B(B).unsqueeze(axis=0)
                raw_out = self.raw_out[b_st:b_ed].unsqueeze(axis=0).cuda(
                    blocking=True)
                raw_grad = self.raw_grad[b_st:b_ed].cuda(blocking=True
                    ) if self.raw_grad is not None else None
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                    
                    # cur_A_interval = tmp_A_interval.tile(repeat_times=[p_ed -
                    #     p_st, 1, 1, 1, 1, 1, 1, 1])
                    # cur_A_interval[:, :, :, :, v:v + 1, :, h:h + 1, :
                    #     ] = A_interval_candidates[p_st:p_ed, :, :, :, v:v +
                    #     1, :, h:h + 1, :]
                    
                    cur_A_interval = tmp_A_interval.repeat_interleave(repeats=p_ed - p_st, axis=0)

                    cur_inter_shape = cur_A_interval.shape
                    A_iinter_shape = A_interval_candidates.shape
                    dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = cur_A_interval.shape
                    v_start = v * dim4 * dim6
                    v_end = (v + 1) * dim4 * dim6
                    cur_A_interval = cur_A_interval.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    A_interval_candidates = A_interval_candidates.reshape([A_iinter_shape[0], dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    cur_A_interval[:, :, :, v_start:v_end, h:h + 1, :] = A_interval_candidates[p_st:p_ed, :, :,v_start:v_end, h:h + 1, :]
                    cur_A_interval = cur_A_interval.reshape(cur_inter_shape)
                    A_interval_candidates = A_interval_candidates.reshape(A_iinter_shape)

                    # cur_A_zero = tmp_A_zerop.tile(repeat_times=[p_ed - p_st,
                    #     1, 1, 1, 1, 1, 1, 1])
                    # cur_A_zero[:, :, :, :, v:v + 1, :, h:h + 1, :
                    #     ] = A_zeropoint_candidates[p_st:p_ed, :, :, :, v:v +
                    #     1, :, h:h + 1, :]
                    
                    cur_A_zero = tmp_A_zerop.repeat_interleave(repeats=p_ed - p_st, axis=0)

                    cur_A_shape = cur_A_zero.shape
                    A_zeropoint_shape = A_zeropoint_candidates.shape
                    dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = cur_A_zero.shape
                    v_start = v * dim4 * dim6
                    v_end = (v + 1) * dim4 * dim6
                    cur_A_zero = cur_A_zero.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    A_zeropoint_candidates = A_zeropoint_candidates.reshape([A_zeropoint_shape[0], dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    cur_A_zero[:, :, :, v_start:v_end, h:h + 1, :] = A_zeropoint_candidates[p_st:p_ed, :, :,v_start:v_end, h:h + 1, :]
                    cur_A_zero = cur_A_zero.reshape(cur_A_shape)
                    A_zeropoint_candidates = A_zeropoint_candidates.reshape(A_zeropoint_shape)

                    A_int = paddle.round(A_pad / cur_A_interval)
                    A_int = paddle.clip(x=A_int + cur_A_zero, min=-self.
                        A_qmax, max=self.A_qmax - 1)
                    A_sim = (A_int - cur_A_zero) * cur_A_interval
                    del A_int, cur_A_interval, cur_A_zero
                    A_sim = A_sim.reshape([p_ed - p_st, -1, tuple(A.shape)[1] +
                        self.pad_groups_A, tuple(A.shape)[2] + self.
                        pad_rows_A, tuple(A.shape)[3] + self.pad_cols_A])
                    A_sim = A_sim[:, :, :tuple(A.shape)[1], :tuple(A.shape)
                        [2], :tuple(A.shape)[3]]
                    out_sim = A_sim @ B_sim
                    similarity = self._get_similarity(raw_out, out_sim,
                        self.metric, raw_grad=raw_grad)
                    if len(tuple(similarity.shape)) > 3:
                        similarity = similarity.mean(axis=[3])
                    similarity = similarity.sum(axis=1, keepdim=True)
                    similarities.append(similarity)
                similarities = paddle.concat(x=similarities, axis=0)
                batch_similarities.append(similarities)
            batch_similarities = paddle.concat(x=batch_similarities, axis=1
                ).sum(axis=1, keepdim=False)
            batch_similarities = paddle.nn.functional.pad(x=
                batch_similarities, pad=[0, self.pad_groups_A],
                pad_from_left_axis=False).reshape([self.eq_n, self.n_G_A, self.
                crb_groups_A]).mean(axis=-1)
            best_index = paddle.argmax(x=batch_similarities, axis=0,
                keepdim=False).reshape([1, 1, -1, 1, 1, 1, 1, 1]).cuda(blocking=True
                )
            
            # tmp_A_interval[:, :, :, :, v:v + 1, :, h:h + 1, :
            #     ] = paddle.take_along_axis(arr=A_interval_candidates[:, :,
            #     :, :, v:v + 1, :, h:h + 1, :], axis=0, indices=best_index,
            #     broadcast=False)
            # tmp_A_zerop[:, :, :, :, v:v + 1, :, h:h + 1, :
            #     ] = paddle.take_along_axis(arr=A_zeropoint_candidates[:, :,
            #     :, :, v:v + 1, :, h:h + 1, :], axis=0, indices=best_index,
            #     broadcast=False)
            
            interval_temp = paddle.take_along_axis(A_interval_candidates[:, :, :, :, v:v+1, :, h:h+1, :], indices=best_index, axis=0)
            ##降低tmp_A_interval的维度,使其可以进行切片处理
            shape = tmp_A_interval.shape
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = shape
            v_start = v * dim4 * dim6
            v_end = (v + 1) * dim4 * dim6
            tmp_A_interval = tmp_A_interval.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            interval_temp = interval_temp.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            tmp_A_interval[:, :, :,  v_start:v_end, h:h+1, :] = interval_temp
            tmp_A_interval = tmp_A_interval.reshape(shape)
        
            zerop_temp = paddle.take_along_axis(A_zeropoint_candidates[:, :, :, :, v:v+1, :, h:h+1, :], indices=best_index, axis=0)
            ##降低tmp_A_zerop的维度,使其可以进行切片处理
            zerop_shape = tmp_A_zerop.shape
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = zerop_shape
            v_start = v * dim4 * dim6
            v_end = (v + 1) * dim4 * dim6
            tmp_A_zerop = tmp_A_zerop.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            zerop_temp = zerop_temp.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            tmp_A_zerop[:, :, :,  v_start:v_end, h:h+1, :] = zerop_temp
            tmp_A_zerop = tmp_A_zerop.reshape(zerop_shape)

            
        self.A_interval = tmp_A_interval.squeeze(axis=0)
        self.A_zerop = tmp_A_zerop.squeeze(axis=0)

    def _search_best_B_interval(self, B_interval_candidates=None):
        """
        Modularization of searching best interval
        """
        tmp_B_interval = self.B_interval.unsqueeze(axis=0)
        if self.B_zerop is None:
            tmp_B_zerop = paddle.zeros_like(x=tmp_B_interval)
        else:
            tmp_B_zerop = self.B_zerop.unsqueeze(axis=0)
        for v, h in product(range(self.n_V_B), range(self.n_H_B)):
            batch_similarities = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input[0][b_st:b_ed].cuda(blocking=True)
                A_sim = self.quant_input_A(A).unsqueeze(axis=0)
                B = self.raw_input[1][b_st:b_ed].cuda(blocking=True)
                B_pad = paddle.nn.functional.pad(x=B, pad=[0, self.
                    pad_cols_B, 0, self.pad_rows_B, 0, self.pad_groups_B],
                    pad_from_left_axis=False).unsqueeze(axis=0).reshape([1, -1,
                    self.n_G_B, self.crb_groups_B, self.n_V_B, self.
                    crb_rows_B, self.n_H_B, self.crb_cols_B])
                B_max = B_pad.amax(axis=[0, 1, 3, 5, 7], keepdim=True).detach()
                B_min = B_pad.amin(axis=[0, 1, 3, 5, 7], keepdim=True).detach()
                new_max = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 
                    1, 1, 1, 1]) * B_max
                new_min = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 
                    1, 1, 1, 1]) * B_min
                new_scale = (new_max - new_min) / float(self.B_qmax * 2 - 1)
                new_scale.clip_(min=1e-08)
                new_zero_point = -self.B_qmax - paddle.round(new_min /
                    new_scale)
                new_zero_point.clip_(min=-self.B_qmax, max=self.B_qmax - 1)
                B_zeropoint_candidates = new_zero_point.cuda(blocking=True)
                B_interval_candidates = new_scale.cuda(blocking=True)
                raw_out = self.raw_out[b_st:b_ed].unsqueeze(axis=0).cuda(
                    blocking=True)
                raw_grad = self.raw_grad[b_st:b_ed].cuda(blocking=True
                    ) if self.raw_grad is not None else None
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                             
                    cur_B_interval = tmp_B_interval.repeat_interleave(repeats=p_ed - p_st, axis=0)   
                    # cur_B_interval[:, :, :, :, v:v + 1, :, h:h + 1, :
                    #     ] = B_interval_candidates[p_st:p_ed, :, :, :, v:v +
                    #     1, :, h:h + 1, :]
                    cur_inter_shape = cur_B_interval.shape
                    B_inter_shape = B_interval_candidates.shape
                    dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = cur_B_interval.shape
                    v_start = v * dim4 * dim6
                    v_end = (v + 1) * dim4 * dim6
                    cur_B_interval = cur_B_interval.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    B_interval_candidates = B_interval_candidates.reshape([B_inter_shape[0], dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    cur_B_interval[:, :, :, v_start:v_end, h:h + 1, :] = B_interval_candidates[p_st:p_ed, :, :,v_start:v_end, h:h + 1, :]
                    cur_B_interval = cur_B_interval.reshape(cur_inter_shape)
                    B_interval_candidates = B_interval_candidates.reshape(B_inter_shape)
                
                    
                    cur_B_zero = tmp_B_zerop.repeat_interleave(repeats=p_ed - p_st, axis=0) 
                    # cur_B_zero[:, :, :, :, v:v + 1, :, h:h + 1, :
                    #     ] = B_zeropoint_candidates[p_st:p_ed, :, :, :, v:v +
                    #     1, :, h:h + 1, :]
                    cur_B_shape = cur_B_zero.shape
                    B_zeropoint_shape = B_zeropoint_candidates.shape
                    dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = cur_B_zero.shape
                    v_start = v * dim4 * dim6
                    v_end = (v + 1) * dim4 * dim6
                    cur_B_zero = cur_B_zero.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    B_zeropoint_candidates = B_zeropoint_candidates.reshape([B_zeropoint_shape[0], dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    cur_B_zero[:, :, :, v_start:v_end, h:h + 1, :] = B_zeropoint_candidates[p_st:p_ed, :, :,v_start:v_end, h:h + 1, :]
                    cur_B_zero = cur_B_zero.reshape(cur_B_shape)
                    B_zeropoint_candidates = B_zeropoint_candidates.reshape(B_zeropoint_shape)
                    
                    B_int = paddle.round(B_pad / cur_B_interval)
                    B_int = paddle.clip(x=B_int + cur_B_zero, min=-self.
                        B_qmax, max=self.B_qmax - 1)
                    B_sim = (B_int - cur_B_zero) * cur_B_interval
                    del B_int, cur_B_interval, cur_B_zero
                    B_sim = B_sim.reshape([p_ed - p_st, -1, tuple(B.shape)[1] +
                        self.pad_groups_B, tuple(B.shape)[2] + self.
                        pad_rows_B, tuple(B.shape)[3] + self.pad_cols_B])
                    B_sim = B_sim[:, :, :tuple(B.shape)[1], :tuple(B.shape)
                        [2], :tuple(B.shape)[3]]
                    out_sim = A_sim @ B_sim
                    similarity = self._get_similarity(raw_out, out_sim,
                        self.metric, raw_grad=raw_grad)
                    if len(tuple(similarity.shape)) > 3:
                        similarity = similarity.mean(axis=[3])
                    similarity = similarity.sum(axis=1, keepdim=True)
                    similarities.append(similarity)
                similarities = paddle.concat(x=similarities, axis=0)
                batch_similarities.append(similarities)
            batch_similarities = paddle.concat(x=batch_similarities, axis=1).sum(axis=1, keepdim=False)
            batch_similarities = paddle.nn.functional.pad(x=batch_similarities, pad=[0, self.pad_groups_B],pad_from_left_axis=False).reshape([self.eq_n, self.n_G_B, self.crb_groups_B]).mean(axis=-1)
            best_index = paddle.argmax(x=batch_similarities, axis=0,keepdim=False).reshape([1, 1, -1, 1, 1, 1, 1, 1]).cuda(blocking=True)
            
            # tmp_B_interval[:, :, :, :, v:v + 1, :, h:h + 1, :
            #     ] = paddle.take_along_axis(arr=B_interval_candidates[:, :,
            #     :, :, v:v + 1, :, h:h + 1, :], axis=0, indices=best_index,
            #     broadcast=False)
            # tmp_B_zerop[:, :, :, :, v:v + 1, :, h:h + 1, :
            #     ] = paddle.take_along_axis(arr=B_zeropoint_candidates[:, :,
            #     :, :, v:v + 1, :, h:h + 1, :], axis=0, indices=best_index,
            #     broadcast=False)
            
            interval_temp = paddle.take_along_axis(B_interval_candidates[:, :, :, :, v:v+1, :, h:h+1, :], indices=best_index, axis=0)
            ##降低tmp_B_interval的维度,使其可以进行切片处理
            shape = tmp_B_interval.shape
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = shape
            v_start = v * dim4 * dim6
            v_end = (v + 1) * dim4 * dim6
            tmp_B_interval = tmp_B_interval.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            interval_temp = interval_temp.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            tmp_B_interval[:, :, :,  v_start:v_end, h:h+1, :] = interval_temp
            tmp_B_interval = tmp_B_interval.reshape(shape)
        
            zerop_temp = paddle.take_along_axis(B_zeropoint_candidates[:, :, :, :, v:v+1, :, h:h+1, :], indices=best_index, axis=0)
            ##降低tmp_B_zerop的维度,使其可以进行切片处理
            zerop_shape = tmp_B_zerop.shape
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = zerop_shape
            v_start = v * dim4 * dim6
            v_end = (v + 1) * dim4 * dim6
            tmp_B_zerop = tmp_B_zerop.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            zerop_temp = zerop_temp.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            tmp_B_zerop[:, :, :,  v_start:v_end, h:h+1, :] = zerop_temp
            tmp_B_zerop = tmp_B_zerop.reshape(zerop_shape)

        self.B_interval = tmp_B_interval.squeeze(axis=0)
        self.B_zerop = tmp_B_zerop.squeeze(axis=0)

    def calibration_step2(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha + i *
            (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(self
            .eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1, 1, 1, 1]
            ) * self.A_interval.unsqueeze(axis=0)
        B_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha + i *
            (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(self
            .eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1, 1, 1, 1]
            ) * self.B_interval.unsqueeze(axis=0)
        for e in range(self.search_round):
            self._search_best_A_interval(A_interval_candidates)
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad


def get_shift_and_sign_for_each_head(x, n=2.0, A_qmax=None):
    sign = paddle.sign(x=x)
    x_abs = paddle.abs(x=x)
    shift = paddle.concat(x=[(-paddle.log(x=x_abs[:, :, i:i + 1, :, :]) /
        paddle.log(x=n[i])) for i in range(tuple(n.shape)[0])], axis=2).round(
        ).clip(min=-A_qmax, max=A_qmax - 1)
    return shift, sign


def fake_quantize_headwise(x, n=2.0, A_qmax=None):
    shift, sign = get_shift_and_sign_for_each_head(x, n=n, A_qmax=A_qmax)
    x_rounded = paddle.concat(x=[(n[i] ** shift[:, :, i:i + 1, :, :] * sign
        [:, :, i:i + 1, :, :]) for i in range(tuple(n.shape)[0])], axis=2)
    return x_rounded


class HeadWiseLogQuant(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, input, n=2, A_qmax=None):
        return fake_quantize_headwise(input, n=n, A_qmax=A_qmax)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def headwise_log_quant(input, n=2, A_qmax=None):
    return HeadWiseLogQuant.apply(input, n, A_qmax)


class SoSPTQSLBatchingQuantMatMul(PTQSLBatchingQuantMatMul):

    def __init__(self, A_bit=8, B_bit=8, mode='raw', metric='L2_norm',
        search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
        n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1,
        init_layerwise=False, split=None):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, metric=metric,
            search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta,
            eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_G_A=n_G_A, n_V_A=
            n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B,
            init_layerwise=init_layerwise)
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        self.A_qmax = 2 ** (self.A_bit - 1)
        self.split = split
        if split != None:
            self.A_interval = self.split / (self.A_qmax - 1)
        self.log_n = None

    def _search_best_A_interval(self, A_interval_candidates=None):
        """
        Modularization of searching best interval
        """
        tmp_A_interval = self.A_interval.unsqueeze(axis=0)
        if self.A_zerop is None:
            tmp_A_zerop = paddle.zeros_like(x=tmp_A_interval)
        else:
            tmp_A_zerop = self.A_zerop.unsqueeze(axis=0)
        for v, h in product(range(self.n_V_A), range(self.n_H_A)):
            batch_similarities = []
            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                A = self.raw_input[0][b_st:b_ed].cuda(blocking=True)
                A_pad = paddle.nn.functional.pad(x=A, pad=[0, self.
                    pad_cols_A, 0, self.pad_rows_A, 0, self.pad_groups_A],
                    pad_from_left_axis=False).unsqueeze(axis=0).reshape([1, -1,
                    self.n_G_A, self.crb_groups_A, self.n_V_A, self.
                    crb_rows_A, self.n_H_A, self.crb_cols_A])
                A_max = A_pad.amax(axis=[0, 1, 3, 5, 7], keepdim=True).detach()
                A_min = A_pad.amin(axis=[0, 1, 3, 5, 7], keepdim=True).detach()
                new_max = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 
                    1, 1, 1, 1]) * A_max
                new_min = paddle.to_tensor(data=[(self.eq_alpha + i * (self
                    .eq_beta - self.eq_alpha) / self.eq_n) for i in range(
                    self.eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 
                    1, 1, 1, 1]) * A_min
                new_scale = (new_max - new_min) / float(self.A_qmax * 2 - 1)
                new_scale.clip_(min=1e-08)
                new_zero_point = -self.A_qmax - paddle.round(new_min /
                    new_scale)
                new_zero_point.clip_(min=-self.A_qmax, max=self.A_qmax - 1)
                A_zeropoint_candidates = new_zero_point.cuda(blocking=True)
                A_interval_candidates = new_scale.cuda(blocking=True)
                B = self.raw_input[1][b_st:b_ed].cuda(blocking=True)
                B_sim = self.quant_input_B(B).unsqueeze(axis=0)
                raw_out = self.raw_out[b_st:b_ed].unsqueeze(axis=0).cuda(
                    blocking=True)
                raw_grad = self.raw_grad[b_st:b_ed].cuda(blocking=True
                    ) if self.raw_grad is not None else None
                similarities = []
                for p_st in range(0, self.eq_n, self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st + self.parallel_eq_n)

                    cur_A_interval = tmp_A_interval.repeat_interleave(repeats=p_ed - p_st, axis=0)
                    cur_inter_shape = cur_A_interval.shape
                    A_iinter_shape = A_interval_candidates.shape
                    dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = cur_A_interval.shape
                    v_start = v * dim4 * dim6
                    v_end = (v + 1) * dim4 * dim6
                    cur_A_interval = cur_A_interval.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    A_interval_candidates = A_interval_candidates.reshape([A_iinter_shape[0], dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    cur_A_interval[:, :, :, v_start:v_end, h:h + 1, :] = A_interval_candidates[p_st:p_ed, :, :,v_start:v_end, h:h + 1, :]
                    cur_A_interval = cur_A_interval.reshape(cur_inter_shape)
                    A_interval_candidates = A_interval_candidates.reshape(A_iinter_shape)
                
                    # cur_A_interval[:, :, :, :, v:v + 1, :, h:h + 1, :
                    #     ] = A_interval_candidates[p_st:p_ed, :, :, :, v:v +
                    #     1, :, h:h + 1, :]

                    cur_A_zero = tmp_A_zerop.repeat_interleave(repeats=p_ed - p_st, axis=0)
                    cur_A_shape = cur_A_zero.shape
                    A_zeropoint_shape = A_zeropoint_candidates.shape
                    dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = cur_A_zero.shape
                    v_start = v * dim4 * dim6
                    v_end = (v + 1) * dim4 * dim6
                    cur_A_zero = cur_A_zero.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    A_zeropoint_candidates = A_zeropoint_candidates.reshape([A_zeropoint_shape[0], dim2, dim3, dim4*dim5*dim6, dim7, dim8])
                    cur_A_zero[:, :, :, v_start:v_end, h:h + 1, :] = A_zeropoint_candidates[p_st:p_ed, :, :,v_start:v_end, h:h + 1, :]
                    cur_A_zero = cur_A_zero.reshape(cur_A_shape)
                    A_zeropoint_candidates = A_zeropoint_candidates.reshape(A_zeropoint_shape)

                    # cur_A_zero[:, :, :, :, v:v + 1, :, h:h + 1, :
                    #     ] = A_zeropoint_candidates[p_st:p_ed, :, :, :, v:v +
                    #     1, :, h:h + 1, :]

                    A_int = paddle.round(A_pad / cur_A_interval)
                    A_int = paddle.clip(x=A_int + cur_A_zero, min=-self.
                        A_qmax, max=self.A_qmax - 1)
                    A_sim = (A_int - cur_A_zero) * cur_A_interval
                    del A_int, cur_A_interval, cur_A_zero
                    A_sim = A_sim.reshape([p_ed - p_st, -1, tuple(A.shape)[1] +
                        self.pad_groups_A, tuple(A.shape)[2] + self.
                        pad_rows_A, tuple(A.shape)[3] + self.pad_cols_A])
                    A_sim = A_sim[:, :, :tuple(A.shape)[1], :tuple(A.shape)
                        [2], :tuple(A.shape)[3]]
                    out_sim = A_sim @ B_sim
                    similarity = self._get_similarity(raw_out, out_sim,
                        self.metric, raw_grad=raw_grad)
                    if len(tuple(similarity.shape)) > 3:
                        similarity = similarity.mean(axis=[3])
                    similarity = similarity.sum(axis=1, keepdim=True)
                    similarities.append(similarity)
                similarities = paddle.concat(x=similarities, axis=0)
                batch_similarities.append(similarities)
            batch_similarities = paddle.concat(x=batch_similarities, axis=1).sum(axis=1, keepdim=False)
            batch_similarities = paddle.nn.functional.pad(x=batch_similarities, pad=[0, self.pad_groups_A],pad_from_left_axis=False).reshape([self.eq_n, self.n_G_A, self.crb_groups_A]).mean(axis=-1)
            best_index = paddle.argmax(x=batch_similarities, axis=0,keepdim=False).reshape([1, 1, -1, 1, 1, 1, 1, 1]).cuda(blocking=True)

            interval_temp = paddle.take_along_axis(A_interval_candidates[:, :, :, :, v:v+1, :, h:h+1, :], indices=best_index, axis=0)
            ##降低tmp_A_interval的维度,使其可以进行切片处理
            shape = tmp_A_interval.shape
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = shape
            v_start = v * dim4 * dim6
            v_end = (v + 1) * dim4 * dim6
            tmp_A_interval = tmp_A_interval.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            interval_temp = interval_temp.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            tmp_A_interval[:, :, :,  v_start:v_end, h:h+1, :] = interval_temp
            tmp_A_interval = tmp_A_interval.reshape(shape)
        
            zerop_temp = paddle.take_along_axis(A_zeropoint_candidates[:, :, :, :, v:v+1, :, h:h+1, :], indices=best_index, axis=0)
            ##降低tmp_A_zerop的维度,使其可以进行切片处理
            zerop_shape = tmp_A_zerop.shape
            dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = zerop_shape
            v_start = v * dim4 * dim6
            v_end = (v + 1) * dim4 * dim6
            tmp_A_zerop = tmp_A_zerop.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            zerop_temp = zerop_temp.reshape([dim1, dim2, dim3, dim4*dim5*dim6, dim7, dim8])
            tmp_A_zerop[:, :, :,  v_start:v_end, h:h+1, :] = zerop_temp
            tmp_A_zerop = tmp_A_zerop.reshape(zerop_shape)

        self.A_interval = tmp_A_interval.squeeze(axis=0)
        self.A_zerop = tmp_A_zerop.squeeze(axis=0)

    def quant_input_A(self, x):
        x = paddle.nn.functional.pad(x=x, pad=[0, self.pad_cols_A, 0, self.
            pad_rows_A, 0, self.pad_groups_A], pad_from_left_axis=False)
        x = x.reshape([-1, self.n_G_A, self.crb_groups_A, self.n_V_A, self.
            crb_rows_A, self.n_H_A, self.crb_cols_A])
        x = ((x / self.A_interval).round_() + self.A_zerop).clip(min=-self.
            A_qmax, max=self.A_qmax - 1)
        x = (x - self.A_zerop).multiply_(y=paddle.to_tensor(self.A_interval))
        x = x.reshape([-1, self.n_G_A * self.crb_groups_A, self.n_V_A * self.
            crb_rows_A, self.n_H_A * self.crb_cols_A])
        x = x[:, :tuple(x.shape)[1] - self.pad_groups_A, :tuple(x.shape)[2] -
            self.pad_rows_A, :tuple(x.shape)[3] - self.pad_cols_A]
        return x

    def calibration_step2(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha + i *
            (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(self
            .eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1, 1, 1, 1
            ]) * self.A_interval.unsqueeze(axis=0)
        B_interval_candidates = paddle.to_tensor(data=[(self.eq_alpha + i *
            (self.eq_beta - self.eq_alpha) / self.eq_n) for i in range(self
            .eq_n + 1)]).cuda(blocking=True).reshape([-1, 1, 1, 1, 1, 1, 1, 1]
            ) * self.B_interval.unsqueeze(axis=0)
        B_zeropoint_candidates = paddle.zeros_like(x=B_interval_candidates)
        for e in range(self.search_round):
            self._search_best_A_interval(A_interval_candidates)
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
