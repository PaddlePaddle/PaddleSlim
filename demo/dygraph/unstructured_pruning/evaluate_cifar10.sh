#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python3.7 evaluate.py \
          --pruned_model="models/model-pruned.pdparams" \
          --data="cifar10"
