#!/usr/bin/env bash

# for single card train
# python tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m paddle.distributed.launch --gpus="4,5,6,7" train_qt.py -c ../ppcls/configs/ImageNet/ResNet/ResNet18_torch.yaml
