python3.7 -m paddle.distributed.launch \
          --selected_gpus="0,1,2,3" \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.55 \
          --lr 0.05 \
          --model MobileNet \
          --pretrained_model "MobileNetV1_pretrained"
