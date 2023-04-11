import argparse
import six
from inspect import isfunction
from types import FunctionType
from typing import Dict
import paddle.vision.models as models

SUPPORT_MODELS: Dict[str, FunctionType] = {}
for _name, _module in models.__dict__.items():
    if isfunction(_module) and 'pretrained' in _module.__code__.co_varnames:
        SUPPORT_MODELS[_name] = _module


def parse_args():
    parser = create_argparse()
    args = parser.parse_args()
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")
    return args


def create_argparse():
    parser = argparse.ArgumentParser("Quantization on ImageNet")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Single Card Minibatch size.", )

    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Whether to use pretrained model.")

    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=True,
        help="Whether to use GPU or not.", )
    parser.add_argument(
        "--model", type=str, default="mobilenet_v1", help="The target model.")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="The learning rate used to fine-tune pruned model.")
    parser.add_argument(
        "--lr_strategy",
        type=str,
        default="piecewise_decay",
        help="The learning rate decay strategy.")
    parser.add_argument(
        "--l2_decay", type=float, default=3e-5, help="The l2_decay parameter.")
    parser.add_argument(
        "--ls_epsilon", type=float, default=0.0, help="Label smooth epsilon.")
    parser.add_argument(
        "--use_pact",
        type=bool,
        default=False,
        help="Whether to use PACT method.")
    parser.add_argument(
        "--ce_test", type=bool, default=False, help="Whether to CE test.")
    parser.add_argument(
        "--onnx_format",
        type=bool,
        default=False,
        help="Whether to export the quantized model with format of ONNX.")
    parser.add_argument(
        "--momentum_rate",
        type=float,
        default=0.9,
        help="The value of momentum_rate.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="The number of total epochs.")
    parser.add_argument(
        "--total_images",
        type=int,
        default=1281167,
        help="The number of total training images.")
    parser.add_argument(
        "--data",
        type=str,
        default="imagenet",
        help="Which data to use. 'cifar10' or 'imagenet'")
    parser.add_argument(
        "--log_period", type=int, default=10, help="Log period in batches.")
    parser.add_argument(
        "--infer_model",
        type=str,
        default="./infer_model/int8_infer",
        help="inference model saved directory.")

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints",
        help="checkpoints directory.")

    parser.add_argument(
        "--step_epochs",
        nargs="+",
        type=int,
        default=[10, 20, 30],
        help="piecewise decay step")
    return parser
