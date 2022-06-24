# 单卡启动
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='MobileNetV1_infer' \
    --model_filename='inference.pdmodel' \
    --params_filename='inference.pdiparams' \
    --save_dir='./save_quant_mobilev1/' \
    --batch_size=128 \
    --config_path='./configs/mobilenetv1_qat_dis.yaml'\
    --input_shape 3 224 224 \
    --image_reader_type='paddle' \
    --data_dir='ILSVRC2012'
    
# 多卡启动    
# python -m paddle.distributed.launch run.py \
#     --model_dir='MobileNetV1_infer' \
#     --model_filename='inference.pdmodel' \
#     --params_filename='inference.pdiparams' \
#     --save_dir='./save_quant_mobilev1/' \
#     --batch_size=128 \
#     --config_path='./configs/mobilenetv1_qat_dis.yaml'\
#     --data_dir='ILSVRC2012' 
    
