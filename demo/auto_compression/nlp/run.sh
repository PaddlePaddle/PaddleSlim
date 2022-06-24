export FLAGS_cudnn_deterministic=True
python run.py \
    --model_type='ppminilm' \
    --model_dir='./all_original_models/AFQMC' \
    --model_filename='infer.pdmodel' \
    --params_filename='infer.pdiparams' \
    --dataset='clue' \
    --save_dir='./save_afqmc_pruned/' \
    --batch_size=16 \
    --max_seq_length=128 \
    --task_name='afqmc' \
    --config_path='./configs/afqmc.yaml' 


