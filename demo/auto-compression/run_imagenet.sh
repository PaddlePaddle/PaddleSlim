python3.7 demo_imagenet.py \
    --model_dir='infermodel_mobilenetv2' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_qat_mbv2/' \
    --devices='cpu' \
    --batch_size=2 \
    --config_path='./configs/CV/mbv2_ptq_hpo.yaml'
