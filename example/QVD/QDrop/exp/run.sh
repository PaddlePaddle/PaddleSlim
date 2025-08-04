#!/bin/bash
PYTHONPATH=../../../../:$PYTHONPATH \
python ../../../main_imagenet.py --data_path data_path \
--arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2  --act_quant --order together --wwq --waq --awq --aaq \
--weight 0.01 --input_prob 0.5 --prob 0.5