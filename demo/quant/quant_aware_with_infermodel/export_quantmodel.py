import os
import sys
import math
import time
import numpy as np
import paddle
import logging
import argparse
import functools

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
sys.path[1] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.quant import quant_post_hpo
from utility import add_arguments, print_arguments
import imagenet_reader as reader
_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('checkpoint_path',       str,  "./inference_model/MobileNet_checkpoints/",  "model dir to save quanted model checkpoints")
add_arg('infermodel_save_path',       str, None,                 "quant infer model save path")

def export(args):
    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()

    assert os.path.exists(args.model_path), "args.model_path doesn't exist"
    assert os.path.isdir(args.model_path), "args.model_path must be a dir"

    def reader_generator(imagenet_reader):
        def gen():
            for i, data in enumerate(imagenet_reader()):
                image, label = data
                image = np.expand_dims(image, axis=0)
                yield image
        return gen

    exe = paddle.static.Executor(place)

    quant_config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'not_quant_pattern': ['skip_quant'],
        'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul']
    }
    train_config={
        "num_epoch": 1000, # training epoch num
        "save_iter_step": 5,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "use_pact": False,
        "quant_model_ckpt_path":checkpoint_path,
        "teacher_model_path": model_path,
        "teacher_model_filename": model_filename,
        "teacher_params_filename": params_filename,
        "model_path": model_path,
        "model_filename": model_filename,
        "params_filename": params_filename,
        "distill_node_pair": ["teacher_fc_0.tmp_0", "fc_0.tmp_0", "teacher_batch_norm_24.tmp_4", "batch_norm_24.tmp_4",
            "teacher_batch_norm_22.tmp_4", "batch_norm_22.tmp_4", "teacher_batch_norm_18.tmp_4", "batch_norm_18.tmp_4",
            "teacher_batch_norm_13.tmp_4", "batch_norm_13.tmp_4", "teacher_batch_norm_5.tmp_4", "batch_norm_5.tmp_4"]
    }
    export_quant_infermodel(exe, place,
        scope=None,
        quant_config=quant_config,
        train_config=train_config,
        checkpoint_path=args.checkpoint_path,
        export_infermodel_path=args.infermodel_save_path)

def main():
    args = parser.parse_args()
    print_arguments(args)
    export(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
