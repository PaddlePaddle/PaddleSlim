# 单卡启动
export CUDA_VISIBLE_DEVICES=0
python3.7 eval.py --save_dir='./save_quant_mobilev1/' --config_path='./configs/MobileNetV1/qat_dis.yaml'   

# 多卡启动  
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch run.py --save_dir='./save_quant_mobilev1/' --config_path='./configs/MobileNetV1/qat_dis.yaml'

