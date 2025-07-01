"""
Patch missing operators and missing modules
"""
import paddle


if 'cumprod' not in paddle.__dict__:
    import numpy as np
    from functools import lru_cache

    @lru_cache()
    def cumprod_mask(axis_length):
        mask = np.ones([axis_length, axis_length]).astype('float32')
        mask = np.tril(mask, k=0)

        return paddle.to_tensor(mask)

    def cumprod(x, axis=None):
        if axis is None:
            x = x.reshape([-1])
            axis = 0
        assert isinstance(axis, int)

        if axis < 0:
            axis = len(x.shape) + axis
        axis_length = x.shape[axis]
        mask = cumprod_mask(axis_length).reshape([*list([1]*axis), -1, axis_length, *list([1]*(len(x.shape)-axis-1))])
        x = x.unsqueeze(axis)
        x = x * mask.detach() + (paddle.ones_like(mask) * (1 - mask)).detach()

        return paddle.prod(x, axis=axis+1)

    paddle.cumprod = cumprod
    paddle.Tensor.cumprod = lambda self, axis=None: cumprod(self, axis)

if 'Subset' not in paddle.io.__dict__:
    class Subset(paddle.io.Dataset):
        def __init__(self, dataset, indices) -> None:
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def __len__(self):
            return len(self.indices)

    paddle.io.Subset = Subset
