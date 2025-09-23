#!/usr/bin/env bash

# for single card eval
# python3.7 tools/eval.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards eval
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" eval_qt.py -c ../ppcls/configs/ImageNet/ResNet/ResNet18_custom.yaml

