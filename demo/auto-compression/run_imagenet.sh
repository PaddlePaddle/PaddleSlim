python3.7 demo_imagenet.py --config_path ./configs/CV/mbv2_ptq_hpo.yaml \
    --model_dir='../auto-compression_origin/MobileNetV2_ssld_infer/' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_qat_mbv2/' \
    --devices='gpu' \
    --batch_size=64 \
