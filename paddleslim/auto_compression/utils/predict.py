import os
import paddle
from paddleslim.analysis import TableLatencyPredictor
from .prune_model import get_sparse_model, get_prune_model
from .fake_ptq import post_quant_fake
import shutil


def predict_compressed_model(model_file, param_file, hardware='SD710'):
    """
    Evaluating the latency of the model under various compression strategies.
    Args:
        model_file(str), param_file(str): The inference model to be compressed.
        hardware(str): Target device.
    Returns:
        latency_dict(dict): The latency latency of the model under various compression strategies.
    """
    latency_dict = {}

    model_filename = model_file.split('/')[-1]
    param_filename = param_file.split('/')[-1]

    predictor = TableLatencyPredictor(hardware)
    latency = predictor.predict(
        model_file=model_file, param_file=param_file, data_type='fp32')
    latency_dict.update({'origin_fp32': latency})
    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    post_quant_fake(
        exe,
        model_dir=os.path.dirname(model_file),
        model_filename=model_filename,
        params_filename=param_filename,
        save_model_path='quant_model',
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
        is_full_quantize=False,
        activation_bits=8,
        weight_bits=8)
    quant_model_file = os.path.join('quant_model', model_filename)
    quant_param_file = os.path.join('quant_model', param_filename)

    latency = predictor.predict(
        model_file=quant_model_file,
        param_file=quant_param_file,
        data_type='int8')
    latency_dict.update({'origin_int8': latency})

    for prune_ratio in [0.3, 0.4, 0.5, 0.6]:
        get_prune_model(
            model_file=model_file,
            param_file=param_file,
            ratio=prune_ratio,
            save_path='prune_model')
        prune_model_file = os.path.join('prune_model', model_filename)
        prune_param_file = os.path.join('prune_model', param_filename)

        latency = predictor.predict(
            model_file=prune_model_file,
            param_file=prune_param_file,
            data_type='fp32')
        latency_dict.update({f'prune_{prune_ratio}_fp32': latency})

        post_quant_fake(
            exe,
            model_dir='prune_model',
            model_filename=model_filename,
            params_filename=param_filename,
            save_model_path='quant_model',
            quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
            is_full_quantize=False,
            activation_bits=8,
            weight_bits=8)
        quant_model_file = os.path.join('quant_model', model_filename)
        quant_param_file = os.path.join('quant_model', param_filename)

        latency = predictor.predict(
            model_file=quant_model_file,
            param_file=quant_param_file,
            data_type='int8')
        latency_dict.update({f'prune_{prune_ratio}_int8': latency})

    for sparse_ratio in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        get_sparse_model(
            model_file=model_file,
            param_file=param_file,
            ratio=sparse_ratio,
            save_path='sparse_model')
        sparse_model_file = os.path.join('sparse_model', model_filename)
        sparse_param_file = os.path.join('sparse_model', param_filename)

        latency = predictor.predict(
            model_file=sparse_model_file,
            param_file=sparse_param_file,
            data_type='fp32')
        latency_dict.update({f'sparse_{sparse_ratio}_fp32': latency})

        post_quant_fake(
            exe,
            model_dir='sparse_model',
            model_filename=model_filename,
            params_filename=param_filename,
            save_model_path='quant_model',
            quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
            is_full_quantize=False,
            activation_bits=8,
            weight_bits=8)
        quant_model_file = os.path.join('quant_model', model_filename)
        quant_param_file = os.path.join('quant_model', param_filename)

        latency = predictor.predict(
            model_file=quant_model_file,
            param_file=quant_param_file,
            data_type='int8')
        latency_dict.update({f'sparse_{sparse_ratio}_int8': latency})

    # Delete temporary model files
    shutil.rmtree('./quant_model')
    shutil.rmtree('./prune_model')
    shutil.rmtree('./sparse_model')
    return latency_dict
