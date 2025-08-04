import paddle
import numpy as np
import pywt
# from paddle.autograd import Function


def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return paddle.concat([x[-n:], x[:-n+end]], axis=0)
    elif dim == 1:
        return paddle.concat([x[:, -n:], x[:, :-n+end]], axis=1)
    elif dim == 2 or dim == -2:
        return paddle.concat([x[:, :, -n:], x[:, :, :-n+end]], axis=2)
    elif dim == 3 or dim == -1:
        return paddle.concat([x[:, :, :, -n:], x[:, :, :, :-n+end]], axis=3)
    

def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))
    

def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))
    

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4

    if not isinstance(g0, paddle.Tensor):
        g0 = paddle.to_tensor(np.copy(np.array(g0).ravel()), dtype='float32', place=lo.place)
    if not isinstance(g1, paddle.Tensor):
        g1 = paddle.to_tensor(np.copy(np.array(g1).ravel()), dtype='float32', place=lo.place)
    
    L = g0.numel().item()
    shape = [1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]

    if g0.shape != tuple(shape):
        g0 = g0.reshape(tuple(shape))
    if g1.shape != tuple(shape):
        g1 = g1.reshape(tuple(shape))

    s = (2, 1) if d == 2 else (1, 2)
    g0 = paddle.concat([g0] * C, axis=0)
    g1 = paddle.concat([g1] * C, axis=0)
    
    if mode == 'per' or mode == 'periodization':
        y = paddle.nn.functional.conv2d_transpose(lo, g0, stride=s, groups=C) + \
            paddle.nn.functional.conv2d_transpose(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:, :, :L-2] = y[:, :, :L-2] + y[:, :, N:N+L-2]
            y = y[:, :, :N]
        else:
            y[:, :, :, :L-2] = y[:, :, :, :L-2] + y[:, :, :, N:N+L-2]
            y = y[:, :, :, :N]
        y = roll(y, shift=1 - L // 2, axis=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (L - 2, 0) if d == 2 else (0, L - 2)
            y = paddle.nn.functional.conv2d_transpose(lo, g0, stride=s, padding=pad, groups=C) + \
                paddle.nn.functional.conv2d_transpose(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError(f"Unknown pad type: {mode}")

    return y


def afb1d(x, h0, h1, mode='zero', dim=-1):
   
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]

    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, paddle.Tensor):
        h0 = paddle.to_tensor(np.copy(np.array(h0).ravel()[::-1]), dtype='float32')
    if not isinstance(h1, paddle.Tensor):
        h1 = paddle.to_tensor(np.copy(np.array(h1).ravel()[::-1]), dtype='float32')

    L = h0.numel().item()
    L2 = L // 2
    shape = [1, 1, 1, 1]
    shape[d] = L

    # If h aren't in the right shape, make them so
    if list(h0.shape) != shape:
        h0 = h0.reshape(shape)
    if list(h1.shape) != shape:
        h1 = h1.reshape(shape)
    h = paddle.concat([h0, h1] * C, axis=0)

    if mode in ['per', 'periodization']:
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = paddle.concat([x, x[:, :, -1:]], axis=2)
            else:
                x = paddle.concat([x, x[:, :, :, -1:]], axis=3)
            N += 1
        x = roll(x, shifts=-L2, axis=d)
        pad = (L - 1, 0) if d == 2 else (0, L - 1)
        lohi = paddle.nn.functional.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N // 2
        if d == 2:
            lohi[:, :, :L2] += lohi[:, :, N2:N2 + L2]
            lohi = lohi[:, :, :N2]
        else:
            lohi[:, :, :, :L2] += lohi[:, :, :, N2:N2 + L2]
            lohi = lohi[:, :, :, :N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = paddle.nn.functional.pad(x, pad)
            pad = (p // 2, 0) if d == 2 else (0, p // 2)
            lohi = paddle.nn.functional.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode in ['symmetric', 'reflect', 'periodic']:
            pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
            x = paddle.nn.functional.pad(x, pad=pad, mode=mode)
            lohi = paddle.nn.functional.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError(f"Unknown pad type: {mode}")

    return lohi



def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
    if g0_row is None:
        g0_row, g1_row = g0_col, g1_col
    else:
        g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)

    g0_col = g0_col.reshape([1, 1, -1, 1])
    g1_col = g1_col.reshape([1, 1, -1, 1])
    g0_row = g0_row.reshape([1, 1, 1, -1])
    g1_row = g1_row.reshape([1, 1, 1, -1])

    return g0_col, g1_col, g0_row, g1_row


def prep_filt_sfb1d(g0, g1, device=None):
    """
    Prepares the filters to be of the right form for the sfb1d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0 (array-like): low pass filter bank
        g1 (array-like): high pass filter bank
        device: which device to put the tensors on to

    Returns:
        (g0, g1)
    """
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()

    t = paddle.get_default_dtype()

    g0 = paddle.to_tensor(g0,  dtype=t, place=device)  
    g1 = paddle.to_tensor(g1, dtype=t, place=device)

    g0 = g0.reshape([1, 1, -1])
    g1 = g1.reshape([1, 1, -1])

    return g0, g1


class SFB2D(paddle.autograd.PyLayer):
    
    @staticmethod
    def forward(ctx, low, highs, g0_row, g1_row, g0_col, g1_col, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(low, highs, g0_row, g1_row, g0_col, g1_col)

        lh, hl, hh = paddle.unbind(highs, axis=2)
        lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
        hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
        y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
          
        low, high, g0_row, g1_row, g0_col, g1_col = ctx.saved_tensor()
        mode = ctx.mode
        if not low.stop_gradient:
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=3)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=2)
            s = dx.shape
            dx = dx.reshape([s[0], -1, 4, s[-2], s[-1]])  
            dlow = dx[:, :, 0].clone()  
            dhigh = dx[:, :, 1:].clone()  
        return dlow, dhigh, None, None, None, None


class AFB2D(paddle.autograd.PyLayer):
    """ Single level 2D wavelet decomposition using PaddlePaddle """

    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(x, h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        ctx.mode = int_to_mode(mode)
        lohi = afb1d(x, h0_row, h1_row, mode=ctx.mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=ctx.mode, dim=2)
        s = y.shape
        y = y.reshape([s[0], -1, 4, s[-2], s[-1]])
        low = paddle.to_tensor(y[:, :, 0])
        highs = paddle.to_tensor(y[:, :, 1:])
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        mode = ctx.mode
        x, h0_row, h1_row, h0_col, h1_col = ctx.saved_tensor()
        if not x.stop_gradient:
            lh, hl, hh = paddle.unbind(highs, axis=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)

            # Adjusting the size if needed
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :, :ctx.shape[-1]]
        return dx, None, None, None, None
    

def prep_filt_afb1d(h0, h1, device=None):

    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    
    t = paddle.get_default_dtype()

    h0 = paddle.to_tensor(h0, dtype=t, place=device).reshape([1, 1, -1])
    h1 = paddle.to_tensor(h1, dtype=t, place=device).reshape([1, 1, -1])
    
    return h0, h1



def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
   

    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    
    if h0_row is None:
        h0_row, h1_row = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)


    h0_col = h0_col.reshape([1, 1, -1, 1])
    h1_col = h1_col.reshape([1, 1, -1, 1])
    h0_row = h0_row.reshape([1, 1, 1, -1])
    h1_row = h1_row.reshape([1, 1, 1, -1])

    return h0_col, h1_col, h0_row, h1_row