import sys
import pickle
import logging
import paddle
from ..common import get_logger
from ..common.load_model import load_inference_model
from ..prune import sensitivity, get_ratios_by_loss

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['analysis_prune']


def get_prune_params(program):
    params = []
    for block in program.blocks:
        for op in block.ops:
            if op.type == 'conv2d' and op.attr('groups') == 1:
                for inp_name in op.input_arg_names:
                    if block.var(inp_name).persistable is True:
                        params.append(inp_name)
    return params


def analysis_prune(eval_function,
                   model_dir,
                   model_filename,
                   params_filename,
                   analysis_file,
                   pruned_ratios,
                   target_loss=None):
    devices = paddle.device.get_device().split(':')[0]
    places = paddle.device._convert_to_place(devices)
    exe = paddle.static.Executor(places)
    [eval_program, feed_target_names, fetch_targets] = (load_inference_model(
        model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        executor=exe))
    params = get_prune_params(eval_program)

    _logger.info("start analysis")
    sens_0 = sensitivity(
        eval_program,
        places,
        params,
        eval_function,
        sensitivities_file=analysis_file,
        eval_args=[exe, feed_target_names, fetch_targets],
        pruned_ratios=pruned_ratios)

    with open(analysis_file, 'rb') as f:
        if sys.version_info < (3, 0):
            sensitivities = pickle.load(f)
        else:
            sensitivities = pickle.load(f, encoding='bytes')

    _logger.info("finish analysis: {}".format(sensitivities))

    if target_loss is not None:
        ratios = get_ratios_by_loss(sensitivities, target_loss)
        _logger.info("you can set prune_params_name: {} in ChannelPrune".format(
            ratios.keys()))
        _logger.info("you can set pruned_ratio: {} in ChannelPrune".format(
            ratios.values()))
