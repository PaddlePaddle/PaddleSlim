data_path="/dataset/ILSVRC2012"
val_dir="val_hapi"
epoch=1
lr=0.0001
batch_size=32
num_workers=3
output_dir=$PWD/output_qat

for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16
do
    python ./src/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --val_dir=${val_dir} \
        --epoch=${epoch} \
        --batch_size=${batch_size} \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=${output_dir} \
        --enable_quant
        #--use_pact
done

echo "\n"
