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
                   target_loss=None,
                   criterion='l1_norm'):
    '''
    Args:
        eval_func(function): The callback function used to evaluate the model. It should accept a instance of `paddle.static.Program` as argument and return a score on test dataset.
        model_dir(str): Directory path to load model. If you want to load onnx model, only set ``model_dir=model.onnx``.
        model_filename(str): Specify model_filename. If you want to load onnx model, model filename should be None.
        params_filename(str): Specify params_filename. If you want to load onnx model, params filename should be None.
        analysis_file(str): The file to save the sensitivities. It will append the latest computed sensitivities into the file. And the sensitivities in the file would not be computed again. This file can be loaded by `pickle` library.
        pruned_ratios(list): The ratios to be pruned.
        criterion(str|function): The criterion used to sort channels for pruning. Currently supports l1_ norm, bn_scale, geometry_median. Default: l1_norm.
    '''

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
        pruned_ratios=pruned_ratios,
        criterion=criterion)

    with open(analysis_file, 'rb') as f:
        if sys.version_info < (3, 0):
            sensitivities = pickle.load(f)
        else:
            sensitivities = pickle.load(f, encoding='bytes')

    _logger.info("finish analysis: {}".format(sensitivities))

    ratios = {}
    if target_loss is not None:
        ratios = get_ratios_by_loss(sensitivities, target_loss)
        _logger.info("you can set prune_params_name: {} in ChannelPrune".format(
            ratios.keys()))
        _logger.info("you can set pruned_ratio: {} in ChannelPrune".format(
            ratios.values()))
    return ratios
