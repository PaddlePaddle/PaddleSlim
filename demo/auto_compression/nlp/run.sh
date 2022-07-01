export CUDA_VISIBLE_DEVICES=0
export FLAGS_cudnn_deterministic=True
python run.py --config_path='./configs/pp-minilm/auto/afqmc.yaml' --save_dir='./save_afqmc_pruned/'

