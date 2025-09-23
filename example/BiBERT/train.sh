export TASK_NAME=SST-2
export TEACHER_PATH=/path/to/SST-2/sst-2_ft_model_6315.pdparams
export LOG_FILENAME=$(date "+%Y-%m-%d-%H-%M-%S")

python my_task_distill.py \
    --model_type bert \
    --student_model_name_or_path $TEACHER_PATH \
    --seed 1029 \
    --weight_decay 0.01 \
    --task_name $TASK_NAME \
    --max_seq_length 64 \
    --batch_size 16 \
    --teacher_model_type bert \
    --teacher_path $TEACHER_PATH \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 10 \
    --output_dir /path/to/output/$TASK_NAME/ \
    --device gpu \
    --pred_distill \
    --query_distill \
    --key_distill \
    --value_distill \
    --intermediate_distill \
    --bi 2>&1 | tee ${LOG_FILENAME}.log