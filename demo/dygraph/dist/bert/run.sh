export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=SST-2
export TEACHER_DIR=/root/work/Distill_PaddleSlim/PaddleNLP/examples/model_compression/tinybert/best_model_610

python3.7 task_distill.py \
    --model_type tinybert \
    --student_model_name_or_path tinybert-6l-768d-v2 \
    --task_name $TASK_NAME \
    --intermediate_distill \
    --max_seq_length 64 \
    --batch_size 32   \
    --T 1 \
    --teacher_model_type bert \
    --teacher_path $TEACHER_DIR \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --logging_steps 10 \
    --save_steps 10 \
    --output_dir ./tmp/$TASK_NAME/ \
    --distill_config ./distill_stage1.yaml \
    --device gpu

