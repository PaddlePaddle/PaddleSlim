# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
from .utils import compute_scales, k_means
from .metrics import mse_loss

__all__ = ['PieceWiseSearch']


class PieceWiseSearch():
    def __init__(self,
                 k_piece=1,
                 bits_length=8,
                 search_piece=False,
                 search_alpha_min=0.2,
                 search_alpha_max=0.8,
                 search_scale_min=1.,
                 search_scale_max=5.,
                 weight_quant_method='abs_max_channel_wise',
                 act_quant_method='abs_max',
                 loss_function=mse_loss):
        '''
        PieceWiseSearch provides to search k_piece, alpha and scale.

        Args:
        k_piece (int): Number of k pieces. Default: 1.
        bits_length (int): Number of bits to quantize the weight. Default: 8.
        search_piece (bool): Whether to search the best k piece. Default: False.
        search_alpha_min (float): Minimum alpha for search. Default: 0.2.
        search_alpha_max (float): Maximum alpha for search. Default: 0.8.
        search_scale_min (float): Minimum scale for search. Default: 1.
        search_scale_max (float): Maximum scale for search. Default: 5.
        weight_quant_method (str): Weight quantization method. Choosen from abs_max, abs_max_channel_wise and avg. Default: abs_max_channel_wise.
        act_quant_method (str): Activation quantization method. Choosen from abs_max, avg. Default: abs_max.
        loss_function (callable): Loss function. Default: mse_loss.
        '''
        self.k_piece = k_piece
        self.bits_length = bits_length
        self.search_piece = search_piece
        self.search_alpha_min = search_alpha_min
        self.search_alpha_max = search_alpha_max
        self.search_scale_min = search_scale_min
        self.search_scale_max = search_scale_max
        self.weight_quant_method = weight_quant_method
        self.act_quant_method = act_quant_method
        self.bnt = (1 << (bits_length - 1)) - 1
        self.loss_function = loss_function

    def search(self, layer_name, sampled_input, act_abs_max, weight):
        act = sampled_input
        act.stop_gradient = True
        print('[smooth search] search input of %s' % layer_name)

        origin_out = paddle.matmul(act, weight)
        w_abs_max = weight.abs().max(axis=-1, keepdim=True)
        rw_abs_max = w_abs_max.reshape(act_abs_max.shape)
        np_act_abs_max = np.array(act_abs_max)
        np_rw_abs_max = np.array(rw_abs_max)

        smooth_scale_out = None
        global_loss = float('inf')
        best_scale = None

        for k_piece in range(1, self.k_piece + 1):
            if not self.search_piece:
                k_piece = self.k_piece
            print('Search {} Piece'.format(k_piece))
            centroids, labels = k_means(act_abs_max, k_piece)
            piece = ['piece_{}'.format(a) for a in range(len(centroids))]
            for i in range(len(centroids)):
                # print('search for piece {}; centroids value is {}'.format(
                #     piece[i], centroids[centroids.argsort()[i]].numpy()))
                alpha = self.search_alpha_min
                alpha_max = self.search_scale_max if self.search_scale_max is not None else self.search_alpha_max
                calibration_loss = float('inf')
                final_alpha = None
                mask_for_search = paddle.where(labels == centroids.argsort()[i],
                                               1., 0.)
                mask_for_ones = paddle.where(mask_for_search == 0., 1., 0.)

                while alpha <= alpha_max:
                    if alpha < 1:
                        alpha += 0.01
                        if alpha >= self.search_alpha_max:
                            alpha = self.search_scale_min
                            if alpha is None:
                                break
                    else:
                        alpha += 0.5

                    alpha = round(alpha, 2)

                    if alpha < 1:
                        s = (np.power(np_act_abs_max, alpha) / np.power(
                            np_rw_abs_max, 1. - alpha)).clip(min=1e-5)
                        s = paddle.to_tensor(s, dtype='float32')
                        smooth_scale = s * mask_for_search
                    else:
                        smooth_scale = alpha * mask_for_search

                    if smooth_scale_out is not None:
                        mask_for_ones_new = paddle.where(
                            smooth_scale_out == 0., 1., 0.)
                        mask_for_ones *= mask_for_ones_new
                        smooth_scale_ = smooth_scale_out + smooth_scale
                        smooth_scale_tmp = smooth_scale_ + mask_for_ones
                    else:
                        smooth_scale_tmp = smooth_scale + mask_for_ones

                    new_act = act / smooth_scale_tmp
                    new_weight = weight * smooth_scale_tmp.reshape(
                        w_abs_max.shape)

                    quant_scale = compute_scales(
                        new_act, method=self.act_quant_method)
                    quant_act = paddle.clip(
                        paddle.round(new_act / quant_scale * self.bnt),
                        -self.bnt - 1, self.bnt)
                    quant_dequant_act = quant_act / self.bnt * quant_scale

                    quant_scale = compute_scales(
                        new_weight, method=self.weight_quant_method)
                    quant_weight = paddle.clip(
                        paddle.round(new_weight / quant_scale * self.bnt),
                        -self.bnt - 1, self.bnt)
                    quant_dequant_weight = quant_weight / self.bnt * quant_scale
                    new_out = paddle.matmul(quant_dequant_act,
                                            quant_dequant_weight)

                    cur_loss = self.loss_function(origin_out, new_out)
                    if cur_loss <= calibration_loss:
                        calibration_loss = cur_loss
                        final_smooth_scale = smooth_scale
                        final_alpha = alpha

                # print("Layer {} Piece {}, loss: {}, alpha : {}".format(
                #     layer_name, piece[i], float(calibration_loss), final_alpha))
                if smooth_scale_out is None:
                    smooth_scale_out = final_smooth_scale
                else:
                    smooth_scale_out += final_smooth_scale

            if calibration_loss < global_loss:
                global_loss = calibration_loss
                best_scale = smooth_scale_out
                if self.search_piece:
                    print('Find Better K-Piece {}'.format(k_piece))
            if not self.search_piece:
                break
        return best_scale
