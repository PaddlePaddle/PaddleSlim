import random
import paddle
import functional


def extrema_fast(src, range_shrink_percentile=0.0, channel_mean=False, sigma=0.0, fast_mode=True):
    return extrema(src, range_shrink_percentile, channel_mean, sigma, fast_mode)


def extrema(src, range_shrink_percentile=0.0, channel_mean=False, sigma=0.0, fast_mode=False):
    if range_shrink_percentile == 0 and sigma == 0 and channel_mean == False:
        mn = src.min()
        mx = src.max()
        return mn, mx
    elif range_shrink_percentile:
        # downsample for fast_mode
        hist_array, mn, mx, mult_factor, offset = tensor_histogram(src, fast_mode=fast_mode)
        if hist_array is None:
            return mn, mx
        
        new_mn_scaled, new_mx_scaled = extrema_hist_search(hist_array, range_shrink_percentile)
        new_mn = (new_mn_scaled / mult_factor) + offset
        new_mx = (new_mx_scaled / mult_factor) + offset

        # take care of floating point inaccuracies that can
        # increase the range (in rare cases) beyond the actual range.
        new_mn = max(mn, new_mn)
        new_mx = min(mx, new_mx)
        return new_mn, new_mx
    elif channel_mean:
        dim = [0, 2, 3] if src.dim() == 4 else None
        mn = paddle.mean(paddle.amin(src, axis=dim, keepdim=False))
        mx = paddle.mean(paddle.amax(src, axis=dim, keepdim=False))
        return mn, mx
    elif sigma:
        mean = paddle.mean(src)
        std = paddle.std(src)
        mn = mean - sigma * std
        mx = mean + sigma * std
        return mn, mx
    else:
        assert False, "unknown extrema computation mode"


# src: paddle.tensor
def tensor_histogram(src, fast_mode=False):
    # downsample for fast_mode
    fast_stride = 2
    fast_stride2 = fast_stride * 2
    if fast_mode and len(src.shape) == 4 and (src.shape[2] > fast_stride2) and (src.shape[3] > fast_stride2):
        r_start = random.randint(0, fast_stride - 1)
        c_start = random.randint(0, fast_stride - 1)
        src = src[..., r_start::fast_stride, c_start::fast_stride]

    mn = src.min()
    mx = src.max()
    if mn == 0 and mx == 0:
        return None, mn, mx, 1.0, 1.0

    # compute range_shrink_percentile based min/max
    # frequency - bincount can only operate on unsigned
    num_bins = 255.0
    cum_freq = 100.0
    offset = mn
    range_val = paddle.abs(mx - mn) + 1e-6
    mult_factor = (num_bins / range_val)
    tensor_int = (src.reshape([-1]) - offset) * mult_factor
    tensor_int = paddle.cast(functional.round_g(tensor_int), dtype=paddle.int32)
    
    # numpy version
    # hist = np.bincount(tensor_int.cpu().numpy())
    # hist_sum = np.sum(hist)
    # hist_array = hist.astype(np.float32) * cum_freq / float(hist_sum)

    # torch version
    # hist = torch.bincount(tensor_int)
    # hist_sum = torch.sum(hist)
    # hist = hist.float() * cum_freq / hist_sum.float()
    # hist_array = hist.cpu().numpy()

    # paddle version
    hist = paddle.bincount(tensor_int)
    hist_sum = paddle.sum(hist)
    hist = paddle.cast(hist, dtype=paddle.float32) * cum_freq / paddle.cast(hist_sum, dtype=paddle.float32)
    hist_array = hist.numpy()

    return hist_array, mn, mx, mult_factor, offset


# this code is not parallelizable. better to pass a numpy array
def extrema_hist_search(hist_array, range_shrink_percentile):
    new_mn_scaled = 0
    new_mx_scaled = len(hist_array) - 1
    hist_sum_left = 0.0
    hist_sum_right = 0.0
    for h_idx in range(len(hist_array)):
        r_idx = len(hist_array) - 1 - h_idx
        hist_sum_left += hist_array[h_idx]
        hist_sum_right += hist_array[r_idx]
        if hist_sum_left < range_shrink_percentile:
            new_mn_scaled = h_idx
        if hist_sum_right < range_shrink_percentile:
            new_mx_scaled = r_idx
    return new_mn_scaled, new_mx_scaled

