model_path=$1
test_samples=1000  # if set as -1, use all test samples
data_path='/dataset/ILSVRC2012/'
batch_size=16

echo "--------eval model: ${model_name}-------------"
python ./src/eval.py \
   --model_path=$model_path \
   --data_dir=${data_path} \
   --test_samples=${test_samples} \
   --batch_size=${batch_size}
