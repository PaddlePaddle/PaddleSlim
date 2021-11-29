data_path="/root/datasets/ILSVRC2012"
quant_batch_num=1
quant_batch_size=16

for model in mobilenet_v2 mobilenet_v1 resnet50 
do
    echo "--------quantize model: ${model}-------------"
    python ./src/ptq.py \
        --data=${data_path} \
        --arch=${model} \
        --fuse=True \
        --quant_batch_num=${quant_batch_num} \
        --quant_batch_size=${quant_batch_size} \
        --output_dir="output_ptq"
done

echo "\n"
