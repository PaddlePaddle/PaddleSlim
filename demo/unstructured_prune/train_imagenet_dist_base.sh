python3.7 -m paddle.distributed.launch \
          --selected_gpus="0,1,2,3" \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --model MobileNet \
          --num_epochs 108 \
          --test_period 5 \
          --model_period 10 \
          --pretrained_model "MobileNetV1_pretrained" \
          --model_path "./models" \
          --step_epochs  71 88 \
          --last_epoch -1 \
          --is_distributed True \
          --pruning_strategy base
