import paddle


def noise_estimation_loss(model,
                          x0: paddle.Tensor,
                          t: paddle.Tensor,
                          e: paddle.Tensor,
                          b: paddle.Tensor, keepdim=False):
    a = (1-b).cumprod(0).index_select(t, 0).reshape((-1, 1, 1, 1))
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.astype('float32'))
    if keepdim:
        return (e - output).square().sum((1, 2, 3))
    else:
        return (e - output).square().sum((1, 2, 3)).mean(0)


loss_registry = {
    'simple': noise_estimation_loss,
}
