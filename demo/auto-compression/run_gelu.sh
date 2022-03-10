python3.7 demo_glue.py --config_path ./configs/NLP/bert_qat_dis.yaml --task 'sst-2' \
    --model_dir='../auto-compression_origin/static_bert_models/' \
    --model_filename='bert.pdmodel' \
    --params_filename='bert.pdiparams' \
    --save_dir='./save_asp_bert/' \
    --devices='gpu' \
    --batch_size=32 \
