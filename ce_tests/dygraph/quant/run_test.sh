data_path='/dataset/ILSVRC2012/'

model_path=$1
use_gpu=$2
ir_optim=False
echo "--------test model: ${model_path}-------------"

python ./src/test.py \
   --model_path=${model_path} \
   --data_dir=${data_path} \
   --test_samples=-1 \
   --batch_size=32 \
   --use_gpu=${use_gpu} \
   --ir_optim=${ir_optim}

echo "\n"
