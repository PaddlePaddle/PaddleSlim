CUDA_VISIBLE_DEVICES=0 python3.7 train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.55 \
          --lr 0.05 \
          --model MobileNet \
          --pretrained_model "MobileNetV1_pretrained" \
