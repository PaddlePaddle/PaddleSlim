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
add_arg('model_path',       str,  "./inference_model/MobileNet/",  "model dir")
add_arg('checkpoint_path',       str,  "./inference_model/MobileNet_checkpoints/",  "model dir to save quanted model checkpoints")
add_arg('model_filename',       str, None,                 "model file name")
add_arg('params_filename',      str, None,                 "params file name")

def eval(exe, place, compiled_test_program, test_feed_names, test_fetch_list):
    val_reader = paddle.batch(reader.val(), batch_size=1)
    image = paddle.static.data(
        name='image', shape=[None, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')

    valid_loader = paddle.io.DataLoader.from_generator(
        feed_list=[image], capacity=512, use_double_buffer=True, iterable=True)
    valid_loader.set_sample_list_generator(val_reader, place)

    results = []
    for batch_id, data in enumerate(val_reader()):
        # top1_acc, top5_acc
        if len(feed_target_names) == 1:
            # eval "infer model", which input is image, output is classification probability
            image = data[0][0].reshape((1, 3, 224, 224))
            label = [[d[1]] for d in data]
            pred = exe.run(val_program,
                           feed={test_feed_names[0]: image},
                           fetch_list=test_fetch_list)
            pred = np.array(pred[0])
            label = np.array(label)
            sort_array = pred.argsort(axis=1)
            top_1_pred = sort_array[:, -1:][:, ::-1]
            top_1 = np.mean(label == top_1_pred)
            top_5_pred = sort_array[:, -5:][:, ::-1]
            acc_num = 0
            for i in range(len(label)):
                if label[i][0] in top_5_pred[i]:
                    acc_num += 1
            top_5 = float(acc_num) / len(label)
            results.append([top_1, top_5])
        else:
            # eval "eval model", which inputs are image and label, output is top1 and top5 accuracy
            image = data[0][0].reshape((1, 3, 224, 224))
            label = [[d[1]] for d in data]
            result = exe.run(compiled_test_program,
                             feed={
                                 test_feed_names[0]: image,
                                 test_feed_names[1]: label
                             },
                             fetch_list=test_fetch_list)
            result = [np.mean(r) for r in result]
            results.append(result)
    result = np.mean(np.array(results), axis=0)
    return result

def quantize(args):
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
    def test_callback(compiled_test_program, feed_names, fetch_list, checkpoint_name):
        ret = eval(exe, place, compiled_test_program, feed_names, fetch_list)
        print("{0} top1_acc/top5_acc= {1}".format(checkpoint_name, ret))

    quant_aware_with_infermodel(
        exe,
        place,
        scope=None,
        train_reader=train_loader,
        quant_config=quant_config,
        train_config=train_config,
        test_callback=test_callback)

def main():
    args = parser.parse_args()
    print_arguments(args)
    quantize(args)


if __name__ == '__main__':
    paddle.enable_static()
    main()
