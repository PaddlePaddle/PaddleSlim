export CUDA_VISIBLE_DEVICES=5

data_path="/dataset/ILSVRC2012"
epoch=1
lr=0.0001
batch_size=32
num_workers=3
output_dir=$PWD/output_models

for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16 
do
    python ./train/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --epoch=${epoch} \
        --batch_size=${batch_size} \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=${output_dir} \
        --enable_quant
        #--use_pact
done
