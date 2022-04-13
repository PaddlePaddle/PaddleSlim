python3.7 demo_glue.py \
    --model_dir='./static_bert_models/' \
    --model_filename='bert.pdmodel' \
    --params_filename='bert.pdiparams' \
    --save_dir='./save_asp_bert/' \
    --devices='cpu' \
    --batch_size=32 \
    --task='sst-2' \
    --config_path='./configs/NLP/bert_asp_dis.yaml'
