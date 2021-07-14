load_model=$1
save_model=$2

python src/save_quant_model.py \
    --load_model_path ${load_model} \
    --save_model_path ${save_model}
