python -m paddle.distributed.launch \
          --selected_gpus="0,1,2,3" \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --model MobileNet \
          --num_epochs 108 \
          --pretrained_model "MobileNetV1_pretrained" \
          --step_epochs  71 88 \
          --initial_ratio 0.15 \
          --pruning_steps 100 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --last_epoch -1 \
          --pruning_strategy gmp
