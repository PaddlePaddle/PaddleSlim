import os
import time
import numpy as np
import paddle
from collections.abc import Iterable

__all__ = ["wrap_dataloader", "get_feed_vars"]


def get_feed_vars(model_dir, model_filename, params_filename):
    """Get feed vars of model.
    """
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(
            model_dir,
            exe,
            model_filename=model_filename,
            params_filename=params_filename))
    return feed_target_names


def wrap_dataloader(dataloader, names):
    """Create a wrapper of dataloader if the data returned by the dataloader is not a dict.
    And the names will be the keys of dict returned by the wrapper.
    """
    if dataloader is None:
        return dataloader
    assert isinstance(dataloader, paddle.io.DataLoader)
    assert len(dataloader) > 0
    data = next(dataloader())
    if isinstance(data, dict):
        return dataloader

    if isinstance(data, Iterable):
        assert len(data) == len(
            names
        ), f"len(data) == len(names), but got len(data): {len(data)} and len(names): {len(names)}"
    else:
        assert len(
            names
        ) == 1, f"The length of name should 1 when data is not Iterable but got {len(names)}"

    def gen():
        for i, data in enumerate(dataloader()):
            if not isinstance(data, Iterable):
                data = [data]
            yield dict((name_, np.array(data_))
                       for name_, data_ in zip(names, data))

    return gen
