#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python evaluate.py \
          --pruned_model="models/model-pruned.pdparams" \
          --data="imagenet"
