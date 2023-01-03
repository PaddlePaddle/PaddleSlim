import paddle


def gram_linear(x):
    x = paddle.Tensor(x).cuda()
    return paddle.matmul(x, paddle.t(x))


def center_gram(gram, unbiased=False):
    gram = gram.cuda()
    if not paddle.allclose(gram, paddle.t(gram)):
        raise ValueError('Input must be a symmetric matrix.')

    means = paddle.mean(gram, 0)
    means -= paddle.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, unbiased=False):
    gram_x = gram_x.cuda()
    gram_y = gram_y.cuda()
    gram_x = center_gram(gram_x, unbiased=unbiased)
    gram_y = center_gram(gram_y, unbiased=unbiased)
    scaled_hsic = paddle.dot(gram_x.reshape([-1]), gram_y.reshape([-1]))
    normalization_x = paddle.linalg.norm(gram_x)
    normalization_y = paddle.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)
