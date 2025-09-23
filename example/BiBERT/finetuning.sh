export TASK_NAME=SST-2
export LOG_FILENAME=$(date "+%Y-%m-%d-%H-%M-%S")

python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir /path/to/output/$TASK_NAME/ \
    --device gpu 2>&1 | tee ${LOG_FILENAME}.log