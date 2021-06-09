python3.7 train_parameters_cifar.py \
    --num_epochs 300 \
    --step_epochs  220 260 \
    --stable_epochs 0 \
    --pruning_epochs 180 \
    --tunning_epochs 120 \
    --ratio_steps_per_epoch 2 \
    --initial_ratio 0.10 \
    --lr 0.01 \
    --lr_strategy linear_decay
