# 单卡启动
export CUDA_VISIBLE_DEVICES=0
python run.py \
    --model_dir='inference_model_usex2paddle' \
    --model_filename='model.pdmodel' \
    --params_filename='model.pdiparams' \
    --save_dir='./save_quant_mobilev1/' \
    --batch_size=128 \
    --config_path='./configs/mobilenetv1_qat_dis.yaml'\
    --input_shape 224 224 3 \
    --image_reader_type='tensorflow' \
    --input_name "input" \
    --data_dir='ILSVRC2012'
    
