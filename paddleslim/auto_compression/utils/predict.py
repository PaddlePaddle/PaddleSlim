import os
import shutil
import paddle
from ...analysis import TableLatencyPredictor
from .prune_model import get_sparse_model, get_prune_model
from .fake_ptq import post_quant_fake
from ...common.load_model import load_inference_model
import platform


def with_variable_shape(model_dir, model_filename=None, params_filename=None):
    """
    Whether the shape of model's input is variable.
    Args:
        path_prefix(str | None): Directory path to save model + model name without suffix.
        model_filename(str): specify model_filename if you don't want to use default name. Default : None.
        params_filename(str): specify params_filename if you don't want to use default name. Default : None.
    Returns:
        bool: Whether the shape of model's input is variable.
    """
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    inference_program, feed_target_names, fetch_targets = load_inference_model(
        model_dir,
        exe,
        model_filename=model_filename,
        params_filename=params_filename)
    for var_ in inference_program.list_vars():
        if var_.name in feed_target_names:
            if var_.shape.count(-1) > 1:
                return True


def predict_compressed_model(executor,
                             places,
                             model_dir,
                             model_filename,
                             params_filename,
                             hardware='SD710'):
    """
    Evaluating the latency of the model under various compression strategies.
    Args:
        model_dir(str): The path of inference model that will be compressed, and
            the model and params that saved by ``paddle.static.io.save_inference_model``
            are under the path.
        model_filename(str, optional):  The name of model file. If parameters
            are saved in separate files, set it as 'None'. Default: 'None'.
        params_filename(str, optional): The name of params file.
            When all parameters are saved in a single file, set it
            as filename. If parameters are saved in separate files,
            set it as 'None'. Default : 'None'.
        hardware(str): Target device.
    Returns:
        latency_dict(dict): The latency latency of the model under various compression strategies.
    """
    local_rank = paddle.distributed.get_rank()
    quant_model_path = 'quant_model_rank_{}_tmp'.format(local_rank)
    prune_model_path = f'prune_model_rank_{local_rank}_tmp'
    sparse_model_path = f'sparse_model_rank_{local_rank}_tmp'

    latency_dict = {}

    model_file = os.path.join(model_dir, model_filename)
    param_file = os.path.join(model_dir, params_filename)

    try:
        predictor = TableLatencyPredictor(hardware)
    except NotImplementedError:
        raise NotImplementedError(
            "Latency predictor cannot used on the platform: {}. That means you can not use latency predictor to select compress strategy automatically, you can set deploy_hardware to None or set compress strategy in the yaml".
            format(platform.system()))
    latency = predictor.predict(
        model_file=model_file, param_file=param_file, data_type='fp32')
    latency_dict.update({'origin_fp32': latency})
    paddle.enable_static()
    post_quant_fake(
        executor,
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename,
        save_model_path=quant_model_path,
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
        is_full_quantize=False,
        activation_bits=8,
        weight_bits=8)
    quant_model_file = os.path.join(quant_model_path, model_filename)
    quant_param_file = os.path.join(quant_model_path, params_filename)

    latency = predictor.predict(
        model_file=quant_model_file,
        param_file=quant_param_file,
        data_type='int8')
    latency_dict.update({'origin_int8': latency})

    for prune_ratio in [0.3, 0.4, 0.5, 0.6]:
        get_prune_model(
            executor,
            places,
            model_file=model_file,
            param_file=param_file,
            ratio=prune_ratio,
            save_path=prune_model_path)
        prune_model_file = os.path.join(prune_model_path, model_filename)
        prune_param_file = os.path.join(prune_model_path, params_filename)

        latency = predictor.predict(
            model_file=prune_model_file,
            param_file=prune_param_file,
            data_type='fp32')
        latency_dict.update({f'prune_{prune_ratio}_fp32': latency})

        post_quant_fake(
            executor,
            model_dir=prune_model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            save_model_path=quant_model_path,
            quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
            is_full_quantize=False,
            activation_bits=8,
            weight_bits=8)
        quant_model_file = os.path.join(quant_model_path, model_filename)
        quant_param_file = os.path.join(quant_model_path, params_filename)

        latency = predictor.predict(
            model_file=quant_model_file,
            param_file=quant_param_file,
            data_type='int8')
        latency_dict.update({f'prune_{prune_ratio}_int8': latency})

    for sparse_ratio in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        get_sparse_model(
            executor,
            places,
            model_file=model_file,
            param_file=param_file,
            ratio=sparse_ratio,
            save_path=sparse_model_path)
        sparse_model_file = os.path.join(sparse_model_path, model_filename)
        sparse_param_file = os.path.join(sparse_model_path, params_filename)

        latency = predictor.predict(
            model_file=sparse_model_file,
            param_file=sparse_param_file,
            data_type='fp32')
        latency_dict.update({f'sparse_{sparse_ratio}_fp32': latency})

        post_quant_fake(
            executor,
            model_dir=sparse_model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            save_model_path=quant_model_path,
            quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
            is_full_quantize=False,
            activation_bits=8,
            weight_bits=8)
        quant_model_file = os.path.join(quant_model_path, model_filename)
        quant_param_file = os.path.join(quant_model_path, params_filename)

        latency = predictor.predict(
            model_file=quant_model_file,
            param_file=quant_param_file,
            data_type='int8')
        latency_dict.update({f'sparse_{sparse_ratio}_int8': latency})

    # NOTE: Delete temporary model files
    if os.path.exists(quant_model_path):
        shutil.rmtree(quant_model_path, ignore_errors=True)
    if os.path.exists(prune_model_path):
        shutil.rmtree(prune_model_path, ignore_errors=True)
    if os.path.exists(sparse_model_path):
        shutil.rmtree(sparse_model_path, ignore_errors=True)
    return latency_dict
