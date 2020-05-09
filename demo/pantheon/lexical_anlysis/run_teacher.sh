#!/bin/bash

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1
export FLAGS_fraction_of_gpu_memory_to_use=0.99

export CUDA_VISIBLE_DEVICES=5,6     # which GPU to use
ERNIE_FINETUNED_MODEL_PATH=./model_finetuned
DATA_PATH=./data/

python -u teacher_ernie.py \
    --ernie_config_path "conf/ernie_config.json" \
    --init_checkpoint "${ERNIE_FINETUNED_MODEL_PATH}" \
    --init_bound 0.1 \
    --vocab_path "conf/vocab.txt" \
    --batch_size 32 \
    --random_seed 0 \
    --num_labels 57 \
    --max_seq_len 128 \
    --test_data "${DATA_PATH}/train.tsv" \
    --label_map_config "./conf/label_map.json" \
    --do_lower_case true \
    --use_cuda true \
    --out_port=5002

