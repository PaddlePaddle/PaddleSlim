CUDA_VISIBLE_DEVICES=0 python3.7 train.py \
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
          --initial_ratio 0.15 \
          --pruning_steps 100 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --is_distributed False \
          --pruning_strategy gmp
