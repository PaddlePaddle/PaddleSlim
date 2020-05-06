#!/bin/bash

export CUDA_VISIBLE_DEVICES=5,6
python -u train_student.py \
    --train_data ./data/train.tsv \
    --test_data ./data/test.tsv \
    --model_save_dir ./teacher_ernie_init_lac_1gru_emb128 \
    --validation_steps 1000 \
    --save_steps 1000 \
    --print_steps 100 \
    --batch_size 32 \
    --epoch 10 \
    --traindata_shuffle_buffer 20000 \
    --word_emb_dim 128 \
    --grnn_hidden_dim 128 \
    --bigru_num 1 \
    --base_learning_rate 1e-3 \
    --emb_learning_rate 2 \
    --crf_learning_rate 0.2 \
    --word_dict_path ./conf/word.dic \
    --label_dict_path ./conf/tag.dic \
    --word_rep_dict_path ./conf/q2b.dic \
    --enable_ce false \
    --use_cuda true \
    --in_address "127.0.0.1:5002"

