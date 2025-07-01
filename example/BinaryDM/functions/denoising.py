import paddle


def compute_alpha(beta, t):
    beta = paddle.concat([paddle.zeros([1]), beta], 0)
    a = (1 - beta).cumprod(0).index_select(t + 1, 0).reshape([-1, 1, 1, 1])
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with paddle.no_grad():
        n = x.shape[0]
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (paddle.ones([n]) * i)
            next_t = (paddle.ones([n]) * j)
            at = compute_alpha(b, t.astype('int64'))
            at_next = compute_alpha(b, next_t.astype('int64'))
            xt = xs[-1]
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * paddle.randn(x.shape) + c2 * et
            xs.append(xt_next)

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with paddle.no_grad():
        n = x.shape[0]
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (paddle.ones([n]) * i)
            next_t = (paddle.ones([n]) * j)
            at = compute_alpha(betas, t.astype('int64'))
            atm1 = compute_alpha(betas, next_t.astype('int64'))
            beta_t = 1 - at / atm1
            x = xs[-1]

            output = model(x, t.astype('float32'))
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = paddle.clip(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e)
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = paddle.randn(x.shape)
            mask = 1 - (t == 0).astype('float32')
            mask = mask.reshape([-1, 1, 1, 1])
            logvar = beta_t.log()
            sample = mean + mask * paddle.exp(0.5 * logvar) * noise
            xs.append(sample)
    return xs, x0_preds
