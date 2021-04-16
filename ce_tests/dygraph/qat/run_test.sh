export CUDA_VISIBLE_DEVICES=7

model_path=$1
test_samples=1000  # if set as -1, use all test samples
data_path='/dataset/ILSVRC2012/'
use_gpu=False
batch_size=10

echo "--------eval model: ${model_name}-------------"
python ./test/eval.py \
   --use_gpu=${use_gpu} \
   --data_dir=${data_path} \
   --test_samples=${test_samples} \
   --model_path=$model_path \
   --batch_size=${batch_size}
