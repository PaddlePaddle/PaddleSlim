#!/usr/bin/env bash
source /home/wsz/anaconda2/bin/activate py27_paddle1.6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#MobileNet v1:
nohup python quanter_test.py \
    --model=MobileNet \
    --pretrained_fp32_model='../../../../pretrain/MobileNetV1_pretrained/' \
    --use_gpu=True \
    --data_dir='/home/ssd8/wsz/tianfei01/traindata/imagenet/' \
    --batch_size=2048 \
    --total_images=1281167 \
    --class_dim=1000 \
    --image_shape=3,224,224 \
    --model_save_dir=output/ \
    --lr_strategy=piecewise_decay \
    --num_epochs=20 \
    --lr=0.0001 \
    --act_quant_type=abs_max \
    --wt_quant_type=abs_max 2>&1 &

#ResNet50:
#python quanter_test.py \
#       --model=ResNet50 \
#       --pretrained_fp32_model=${pretrain_dir}/ResNet50_pretrained \
#       --use_gpu=True \
#       --data_dir=${data_dir} \
#       --batch_size=128 \
#       --total_images=1281167 \
#       --class_dim=1000 \
#       --image_shape=3,224,224 \
#       --model_save_dir=output/ \
#       --lr_strategy=piecewise_decay \
#       --num_epochs=20 \
#       --lr=0.0001 \
#       --act_quant_type=abs_max \
#       --wt_quant_type=abs_max
