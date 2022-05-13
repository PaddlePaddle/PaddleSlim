export FLAGS_cudnn_deterministic=True
export PYTHONPATH=/workspace/xuxu/auto_compress/add_demo/PaddleSlim/:$PYTHONPATH
export PYTHONPATH=/workspace/xuxu/PaddleNLP/:$PYTHONPATH
python3.7 run.py \
    --model_type='ppminilm' \
    --model_dir='./afqmc_base/' \
    --model_filename='inference.pdmodel' \
    --params_filename='inference.pdiparams' \
    --dataset='clue' \
    --save_dir='./save_afqmc_pruned/' \
    --batch_size=16 \
    --max_seq_length=128 \
    --task_name='afqmc' \
    --config_path='./configs/afqmc.yaml' 


