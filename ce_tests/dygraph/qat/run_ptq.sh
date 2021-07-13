data_path="/dataset/ILSVRC2012"
quant_batch_num=10
quant_batch_size=10

for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16 
do
    echo "--------quantize model: ${model}-------------"
    python ./src/ptq.py \
        --data=${data_path} \
        --arch=${model} \
        --quant_batch_num=${quant_batch_num} \
        --quant_batch_size=${quant_batch_size} \
        --output_dir="output_ptq"
done

echo "\n"
