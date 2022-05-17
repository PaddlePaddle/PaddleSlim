python -m paddle.distributed.launch run.py \
    --model_dir='MobileNetV1_infer' \
    --model_filename='inference.pdmodel' \
    --params_filename='inference.pdiparams' \
    --save_dir='./save_quant_mobilev1/' \
    --batch_size=128 \
    --config_path='./configs/mobilev1.yaml'\
    --data_dir='/workspace/dataset/ILSVRC2012/' 
    
