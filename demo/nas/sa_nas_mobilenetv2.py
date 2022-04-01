#!/usr/bin/env bash
##################
#bash slim_ci_demo_all_case.sh $5 $6;

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo -e "\033[31m ${log_path}/FAIL_$2 \033[0m"
    echo "fail log as follow"
    cat  ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo -e "\033[32m ${log_path}/SUCCESS_$2 \033[0m"
    cat  ${log_path}/SUCCESS_$2.log
fi
}

catchException() {
  echo $1 failed due to exception >> FAIL_Exception.log
}

cudaid1=$1;
cudaid2=$2;
echo "cudaid1,cudaid2", ${cudaid1}, ${cudaid2}
export CUDA_VISIBLE_DEVICES=${cudaid1}
#分布式log输出方式
export PADDLE_LOG_LEVEL=debug

export FLAGS_fraction_of_gpu_memory_to_use=0.98
# data PaddleSlim/demo/data/ILSVRC2012
cd ${slim_dir}/demo
if [ -d "data" ];then
    rm -rf data
fi
wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
tar xf ILSVRC2012_data_demo.tar.gz
mv ILSVRC2012_data_demo data
# download pretrain model
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
pre_models="MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd"
if [ -d "pretrain" ];then
    rm -rf pretrain
fi
mkdir pretrain && cd pretrain
for model in ${pre_models}
do
    if [ ! -f ${model} ]; then
        wget -q ${root_url}/${model}_pretrained.tar
        tar xf ${model}_pretrained.tar
    fi
done

# 1 dist
demo_distillation_01(){
cd ${slim_dir}/demo/distillation || catchException demo_distillation
if [ -d "output" ];then
    rm -rf output
fi
export CUDA_VISIBLE_DEVICES=${cudaid1}
python distill.py --num_epochs 1 --save_inference True >${log_path}/demo_distillation_ResNet50_vd_T 2>&1
print_info $? demo_distillation_ResNet50_vd_T

}

demo_distillation_02(){
cd ${slim_dir}/demo/distillation || catchException demo_distillation
if [ -d "output" ];then
    rm -rf output
fi

export CUDA_VISIBLE_DEVICES=${cudaid1}
python distill.py --num_epochs 1 --batch_size 64 --save_inference True \
--model ResNet50 --teacher_model ResNet101_vd \
--teacher_pretrained_model ../pretrain/ResNet101_vd_pretrained >${log_path}/demo_distillation_ResNet101_vd_ResNet50_T 2>&1
print_info $? demo_distillation_ResNet101_vd_ResNet50_T

python distill.py --num_epochs 1 --batch_size 64 --save_inference True \
--model MobileNetV2_x0_25 --teacher_model MobileNetV2 \
--teacher_pretrained_model ../pretrain/MobileNetV2_pretrained >${log_path}/demo_distillation_MobileNetV2_MobileNetV2_x0_25_T 2>&1
print_info $? demo_distillation_MobileNetV2_MobileNetV2_x0_25_T
}

demo_deep_mutual_learning(){
cd ${slim_dir}/demo/deep_mutual_learning || catchException demo_deep_mutual_learning
export CUDA_VISIBLE_DEVICES=${cudaid1}
model=dml_mv1_mv1_gpu1
CUDA_VISIBLE_DEVICES=${cudaid1}
python dml_train.py --epochs 1 >${log_path}/${model} 2>&1
print_info $? ${model}
model=dml_mv1_res50_gpu1
CUDA_VISIBLE_DEVICES=${cudaid1}
python dml_train.py --models='mobilenet-resnet50' --batch_size 128 --epochs 1 >${log_path}/${model} 2>&1
print_info $? ${model}
}

all_distillation(){ # 大数据 5个模型
    demo_distillation_01   # 3
    #demo_distillation_02
    #demo_deep_mutual_learning   # 2
}
# 2.1 quant/quant_aware 使用小数据集即可
demo_quant_quant_aware(){
cd ${slim_dir}/demo/quant/quant_aware || catchException demo_quant_quant_aware
if [ -d "output" ];then
    rm -rf output
fi
export CUDA_VISIBLE_DEVICES=${cudaid1}
# 2.1版本时默认BS=256会报显存不足，故暂时修改成128
python train.py --model MobileNet --pretrained_model ../../pretrain/MobileNetV1_pretrained \
--checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --batch_size 128 >${log_path}/demo_quant_quant_aware_v1 2>&1
print_info $? demo_quant_quant_aware_v1

export CUDA_VISIBLE_DEVICES=${cudaid1}
python train.py --model ResNet34 \
--pretrained_model ../../pretrain/ResNet34_pretrained \
--checkpoint_dir ./output/ResNet34 --num_epochs 1 >${log_path}/demo_quant_quant_aware_ResNet34_T 2>&1
print_info $? demo_quant_quant_aware_ResNet34_T
}
# 2.2 quant/quant_embedding
demo_quant_quant_embedding(){
cd ${slim_dir}/demo/quant/quant_embedding || catchException demo_quant_quant_embedding
export CUDA_VISIBLE_DEVICES=${cudaid1}
# 先使用word2vec的demo数据进行一轮训练，比较量化前infer结果同量化后infer结果different
if [ -d "data" ];then
    rm -rf data
fi
wget -q https://sys-p0.bj.bcebos.com/slim_ci/word_2evc_demo_data.tar.gz --no-check-certificate
tar xf word_2evc_demo_data.tar.gz
mv word_2evc_demo_data data
if [ -d "v1_cpu5_b100_lr1dir" ];then
    rm -rf v1_cpu5_b100_lr1dir
fi
OPENBLAS_NUM_THREADS=1 CPU_NUM=5 python train.py --train_data_dir data/convert_text8 \
--dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu5_b100_lr1dir \
 --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >${log_path}/quant_em_word2vec_T 2>&1
print_info $? quant_em_word2vec_T
# 量化前infer
python infer.py --infer_epoch --test_dir data/test_mid_dir \
--dict_path data/test_build_dict_word_to_id_ \
--batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  \
--start_index 0 --last_index 0 >${log_path}/quant_em_infer1 2>&1
print_info $? quant_em_infer1
# 量化后infer
python infer.py --infer_epoch --test_dir data/test_mid_dir \
--dict_path data/test_build_dict_word_to_id_ \
--batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 \
--last_index 0 --emb_quant True >${log_path}/quant_em_infer2 2>&1
print_info $? quant_em_infer2
}
# 2.3 quan_post # 小数据集
demo_quant_quant_post(){
# 20210425 新增4种离线量化方法
cd ${slim_dir}/demo/quant/quant_post || catchException demo_quant_quant_post
export CUDA_VISIBLE_DEVICES=${cudaid1}
# 1 导出模型
python export_model.py --model "MobileNet" --pretrained_model ../../pretrain/MobileNetV1_pretrained \
--data imagenet >${log_path}/st_quant_post_v1_export 2>&1
print_info $? st_quant_post_v1_export
# 量化前eval
python eval.py --model_path ./inference_model/MobileNet --model_name model \
--params_name weights >${log_path}/st_quant_post_v1_eval1 2>&1
print_info $? st_quant_post_v1_eval1

# 3 离线量化
# 4 量化后eval
for algo in hist avg mse
do
## 不带bc 离线量化
echo "quant_post train no bc " ${algo}
python quant_post.py --model_path ./inference_model/MobileNet \
--save_path ./quant_model/${algo}/MobileNet \
--model_filename model --params_filename weights --algo ${algo} >${log_path}/st_quant_post_v1_T_${algo} 2>&1
print_info $? st_quant_post_v1_T_${algo}
# 量化后eval
echo "quant_post eval no bc " ${algo}
python eval.py --model_path ./quant_model/${algo}/MobileNet --model_name __model__ \
--params_name __params__ > ${log_path}/st_quant_post_${algo}_eval2 2>&1
print_info $? st_quant_post_${algo}_eval2

# 带bc参数的 离线量化
echo "quant_post train bc " ${algo}
python quant_post.py --model_path ./inference_model/MobileNet \
--save_path ./quant_model/${algo}_bc/MobileNet \
--model_filename model --params_filename weights \
--algo ${algo} --bias_correction True >${log_path}/st_quant_post_T_${algo}_bc 2>&1
print_info $? st_quant_post_T_${algo}_bc

# 量化后eval
echo "quant_post eval bc " ${algo}
python eval.py --model_path ./quant_model/${algo}_bc/MobileNet --model_name __model__ \
--params_name __params__ > ${log_path}/st_quant_post_${algo}_bc_eval2 2>&1
print_info $? st_quant_post_${algo}_bc_eval2

done
}

# 2.3 quant_post_hpo # 小数据集
demo_quant_quant_post_hpo(){

cd ${slim_dir}/demo/quant/quant_post_hpo || catchException demo_quant_quant_post_hpo
export CUDA_VISIBLE_DEVICES=${cudaid1}
# 1.导出模型
python ../quant_post/export_model.py \
--model "MobileNet" \
--pretrained_model ../../pretrain/MobileNetV1_pretrained \
--data imagenet > ${log_path}/st_quant_post__hpo_v1_export 2>&1
print_info $? st_quant_post__hpo_v1_export
# 2. quant_post_hpo 设置max_model_quant_count=2
python quant_post_hpo.py  \
--use_gpu=True     \
--model_path="./inference_model/MobileNet/"   \
--save_path="./inference_model/MobileNet_quant/"   \
--model_filename="model"    \
--params_filename="weights"  \
--max_model_quant_count=2 > ${log_path}/st_quant_post_hpo 2>&1
print_info $? st_quant_post_hpo
# 3. 量化后eval
python ../quant_post/eval.py \
--model_path ./inference_model/MobileNet_quant \
--model_name __model__ \
--params_name __params__ > ${log_path}/st_quant_post_hpo_eval 2>&1
print_info $? st_quant_post_hpo_eval

}

#2.4
demo_quant_pact_quant_aware(){
cd ${slim_dir}/demo/quant/pact_quant_aware || catchException demo_quant_pact_quant_aware
export CUDA_VISIBLE_DEVICES=${cudaid1}
# 普通量化,使用小数据集即可
# 2.1版本时默认BS=128 会报显存不足，故暂时修改成64
python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 --lr 0.0001 --use_pact False --batch_size 128 >${log_path}/demo_quant_pact_quant_aware_v3_nopact 2>&1
print_info $? demo_quant_pact_quant_aware_v3_nopact
python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 --lr 0.0001 --use_pact True --batch_size 64 --lr_strategy=piecewise_decay \
--step_epochs 2 --l2_decay 1e-5 >${log_path}/demo_quant_pact_quant_aware_v3 2>&1
print_info $? demo_quant_pact_quant_aware_v3
# load
python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 2 --lr 0.0001 --use_pact True --batch_size 64 --lr_strategy=piecewise_decay \
--step_epochs 20 --l2_decay 1e-5 \
--checkpoint_dir ./output/MobileNetV3_large_x1_0/0 \
--checkpoint_epoch 0 >${log_path}/demo_quant_pact_quant_aware_v3_load 2>&1
print_info $? demo_quant_pact_quant_aware_v3_load
}

# 2.5
demo_dygraph_quant(){
cd ${slim_dir}/demo/dygraph/quant || catchException demo_dygraph_quant
CUDA_VISIBLE_DEVICES=${cudaid1} python train.py --model='mobilenet_v1' \
--pretrained_model '../../pretrain/MobileNetV1_pretrained' \
--num_epochs 1 \
--batch_size 128 \
> ${log_path}/dy_quant_v1_gpu1 2>&1
print_info $? dy_quant_v1_gpu1
# dy_pact_v3
CUDA_VISIBLE_DEVICES=${cudaid1}  python train.py  --lr=0.001 \
--batch_size 128 \
--use_pact=True --num_epochs=1 --l2_decay=2e-5 --ls_epsilon=0.1 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 > ${log_path}/dy_pact_quant_v3_gpu1 2>&1
print_info $? dy_pact_quant_v3_gpu1
# 多卡训练，以0到3号卡为例
CUDA_VISIBLE_DEVICES=${cudaid2}  python -m paddle.distributed.launch \
train.py  --lr=0.001 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--use_pact=True --num_epochs=1 \
--l2_decay=2e-5 \
--ls_epsilon=0.1 \
--batch_size=128 \
--model_save_dir output > ${log_path}/dy_pact_quant_v3_gpu4 2>&1
print_info $? dy_pact_quant_v3_gpu4
}
# 2.6
ce_tests_dygraph_qat(){
cd ${slim_dir}/ce_tests/dygraph/quant || catchException ce_tests_dygraph_qat
ln -s ${slim_dir}/demo/data/ILSVRC2012
test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=16
epoch=1
lr=0.0001
num_workers=1
output_dir=$PWD/output_models
for model in mobilenet_v1
do
#    if [ $1 == nopact ];then
        # 1 quant train
        echo "------1 nopact train--------", ${model}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        python ./src/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --epoch=${epoch} \
        --batch_size=32 \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=${output_dir} \
        --enable_quant > qat_${model}_gpu1_nw1 2>&1
        # 2 eval before save quant
        echo "--------2 eval before save quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./output_models/quant_dygraph/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > eval_before_save_${model} 2>&1
        # 3 CPU上部署量化模型,需要使用`test/save_quant_model.py`脚本进行模型转换。
        echo "--------3 save_nopact_quant_model-------------", ${model}
        python src/save_quant_model.py \
          --load_model_path output_models/quant_dygraph/${model} \
          --save_model_path int8_models/${model} > save_quant_${model} 2>&1
        # 4
        echo "--------4 CPU eval after save nopact quant -------------", ${model}
        export CUDA_VISIBLE_DEVICES=
        python ./src/eval.py \
        --model_path=./int8_models/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > cpu_eval_after_save_${model} 2>&1
#    elif [ $1 == pact ];then
    # 1 pact quant train
        echo "------1 pact train--------", ${model}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        python ./src/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --epoch=${epoch} \
        --batch_size=32 \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=$PWD/output_models_pact/ \
        --enable_quant \
        --use_pact > pact_qat_${model}_gpu1_nw1 2>&1
        # 2 eval before save quant
        echo "--------2 eval before save pact quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./output_models_pact/quant_dygraph/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > eval_before_pact_save_${model} 2>&1
        echo "--------3  save pact quant -------------", ${model}
        python src/save_quant_model.py \
          --load_model_path output_models_pact/quant_dygraph/${model} \
          --save_model_path int8_models_pact/${model} > save_pact_quant_${model} 2>&1
        echo "--------4 CPU eval after save pact quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./int8_models_pact/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > cpu_eval_after_pact_save_${model} 2>&1
#    fi

done
}

ce_tests_dygraph_qat(){
cd ${slim_dir}/ce_tests/dygraph/quant || catchException ce_tests_dygraph_qat4
ln -s ${slim_dir}/demo/data/ILSVRC2012
test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=16
epoch=1
lr=0.0001
num_workers=1
output_dir=$PWD/output_models
for model in mobilenet_v1
#for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16
do

#    if [ $1 == nopact ];then
        # 1 quant train
        echo "------1 nopact train--------", ${model}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        python ./src/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --epoch=${epoch} \
        --batch_size=32 \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=${output_dir} \
        --enable_quant > qat_${model}_gpu1_nw1 2>&1
        # 2 eval before save quant
        echo "--------2 eval before save quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./output_models/quant_dygraph/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > eval_before_save_${model} 2>&1
        # 3 CPU上部署量化模型,需要使用`test/save_quant_model.py`脚本进行模型转换。
        echo "--------3 save_nopact_quant_model-------------", ${model}
        python src/save_quant_model.py \
          --load_model_path output_models/quant_dygraph/${model} \
          --save_model_path int8_models/${model} > save_quant_${model} 2>&1
        # 4
        echo "--------4 CPU eval after save nopact quant -------------", ${model}
        export CUDA_VISIBLE_DEVICES=
        python ./src/eval.py \
        --model_path=./int8_models/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > cpu_eval_after_save_${model} 2>&1
#    elif [ $1 == pact ];then
    # 1 pact quant train
        echo "------1 pact train--------", ${model}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        python ./src/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --epoch=${epoch} \
        --batch_size=32 \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=$PWD/output_models_pact/ \
        --enable_quant \
        --use_pact > pact_qat_${model}_gpu1_nw1 2>&1
        # 2 eval before save quant
        echo "--------2 eval before save pact quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./output_models_pact/quant_dygraph/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > eval_before_pact_save_${model} 2>&1
        echo "--------3  save pact quant -------------", ${model}
        python src/save_quant_model.py \
          --load_model_path output_models_pact/quant_dygraph/${model} \
          --save_model_path int8_models_pact/${model} > save_pact_quant_${model} 2>&1
        echo "--------4 CPU eval after save pact quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./int8_models_pact/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > cpu_eval_after_pact_save_${model} 2>&1
#    fi

done
}

ce_tests_dygraph_ptq(){
cd ${slim_dir}/ce_tests/dygraph/quant || catchException ce_tests_dygraph_ptq4
ln -s ${slim_dir}/demo/data/ILSVRC2012
test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=32
epoch=1
output_dir="./output_ptq"
quant_batch_num=10
quant_batch_size=10
for model in mobilenet_v1
#for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16

do
    echo "--------quantize model: ${model}-------------"
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    # save ptq quant model
    python ./src/ptq.py \
        --data=${data_path} \
        --arch=${model} \
        --quant_batch_num=${quant_batch_num} \
        --quant_batch_size=${quant_batch_size} \
        --output_dir=${output_dir} > ${log_path}/ptq_${model} 2>&1
        print_info $? ptq_${model}

    echo "-------- eval fp32_infer model -------------", ${model}
    python ./src/test.py \
        --model_path=${output_dir}/${model}/fp32_infer \
        --data_dir=${data_path} \
        --batch_size=${batch_size} \
        --use_gpu=True \
        --test_samples=${test_samples} \
        --ir_optim=False > ${log_path}/ptq_eval_fp32_${model} 2>&1
        print_info $? ptq_eval_fp32_${model}

    echo "-------- eval int8_infer model -------------", ${model}
    python ./src/test.py \
        --model_path=${output_dir}/${model}/int8_infer \
        --data_dir=${data_path} \
        --batch_size=${batch_size} \
        --use_gpu=False \
        --test_samples=${test_samples} \
        --ir_optim=False > ${log_path}/ptq_eval_int8_${model} 2>&1
        print_info $? ptq_eval_int8_${model}

done
}

#用于更新release分支下无ce_tests_dygraph_ptq case；release分支设置is_develop="False"
is_develop="True"

all_quant(){ # 10个模型
    if [ "${is_develop}" == "True" ];then
      #ce_tests_dygraph_ptq4
      ce_tests_dygraph_ptq
    fi
    demo_quant_quant_aware    # 2个模型
    demo_quant_quant_embedding  # 1个模型
    demo_quant_quant_post   # 4个策略
    demo_dygraph_quant    # 2个模型
    demo_quant_pact_quant_aware  # 1个模型
    ce_tests_dygraph_qat  # 4个模型
    #ce_tests_dygraph_qat4
    demo_quant_quant_post_hpo
}

# 3 prune
demo_prune(){
cd ${slim_dir}/demo/prune  || catchException demo_prune
# 3.1 P0 prune

if [ -d "models" ];then
    rm -rf models
fi
export CUDA_VISIBLE_DEVICES=${cudaid1}
python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" \
--pretrained_model ../pretrain/MobileNetV1_pretrained/ --num_epochs 1 >${log_path}/prune_v1_T 2>&1
print_info $? prune_v1_T

#3.2 prune_fpgm
# slim_prune_fpgm_v1_T
# export CUDA_VISIBLE_DEVICES=${cudaid1}
# python train.py \
#     --model="MobileNet" \
#     --pretrained_model="../pretrain/MobileNetV1_pretrained" \
#     --data="imagenet" \
#     --pruned_ratio=0.3125 \
#     --lr=0.1 \
#     --num_epochs=1 \
#     --test_period=1 \
#     --step_epochs 30 60 90\
#     --l2_decay=3e-5 \
#     --lr_strategy="piecewise_decay" \
#     --criterion="geometry_median" \
#     --model_path="./fpgm_mobilenetv1_models" \
#     --save_inference True  >${log_path}/slim_prune_fpgm_v1_T 2>&1
# print_info $? slim_prune_fpgm_v1_T

#slim_prune_fpgm_v2_T
export CUDA_VISIBLE_DEVICES=${cudaid1}
#v2 -50%
python train.py \
    --model="MobileNetV2" \
    --pretrained_model="../pretrain/MobileNetV2_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.325 \
    --lr=0.001 \
    --num_epochs=2 \
    --test_period=1 \
    --step_epochs 30 60 80 \
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./output/fpgm_mobilenetv2_models" \
    --save_inference True >${log_path}/slim_prune_fpgm_v2_T 2>&1
print_info $? slim_prune_fpgm_v2_T
python eval.py --model "MobileNetV2" --data "imagenet" \
--model_path "./output/fpgm_mobilenetv2_models/0" >${log_path}/slim_prune_fpgm_v2_eval 2>&1
print_info $? slim_prune_fpgm_v2_eval
# ResNet34 -50
# export CUDA_VISIBLE_DEVICES=${cudaid1}
# python train.py \
#     --model="ResNet34" \
#     --pretrained_model="../pretrain/ResNet34_pretrained" \
#     --data="imagenet" \
#     --pruned_ratio=0.3125 \
#     --lr=0.001 \
#     --num_epochs=2 \
#     --test_period=1 \
#     --step_epochs 30 60 \
#     --l2_decay=1e-4 \
#     --lr_strategy="piecewise_decay" \
#     --criterion="geometry_median" \
#     --model_path="./output/fpgm_resnet34_50_models" \
#     --save_inference True >${log_path}/slim_prune_fpgm_resnet34_50_T 2>&1
print_info $? slim_prune_fpgm_resnet34_50_T
python eval.py --model "ResNet34" --data "imagenet" \
--model_path "./output/fpgm_resnet34_50_models/0" >${log_path}/slim_prune_fpgm_resnet34_50_eval 2>&1
print_info $? slim_prune_fpgm_resnet34_50_eval
# ResNet34 -42 slim_prune_fpgm_resnet34_42_T
cd ${slim_dir}/demo/prune
export CUDA_VISIBLE_DEVICES=${cudaid1}
python train.py \
    --model="ResNet34" \
    --pretrained_model="../pretrain/ResNet34_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --test_period=1 \
    --lr_strategy="cosine_decay" \
    --criterion="geometry_median" \
    --model_path="./output/fpgm_resnet34_025_120_models" \
    --save_inference True >${log_path}/slim_prune_fpgm_resnet34_42_T 2>&1
print_info $? slim_prune_fpgm_resnet34_42_T
python eval.py --model "ResNet34" --data "imagenet" \
--model_path "./output/fpgm_resnet34_025_120_models/0" >${log_path}/slim_prune_fpgm_resnet34_42_eval 2>&1
print_info $? slim_prune_fpgm_resnet34_42_eval
# 3.3 prune ResNet50
export CUDA_VISIBLE_DEVICES=${cudaid1}
# 2.1版本时默认BS=256 会报显存不足，故暂时修改成128
python train.py --model ResNet50 --pruned_ratio 0.31 --data "imagenet" \
--save_inference True --pretrained_model ../pretrain/ResNet50_pretrained \
--num_epochs 1 --batch_size 128 >${log_path}/prune_ResNet50_T 2>&1
print_info $? prune_ResNet50_T
}

# 3.4 dygraph_prune
#dy_prune_ResNet34_f42
demo_dygraph_pruning(){
cd ${slim_dir}/demo/dygraph/pruning  || catchException demo_dygraph_pruning
ln -s ${slim_dir}/demo/data data
CUDA_VISIBLE_DEVICES=${cudaid1} python train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=1 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" >${log_path}/dy_prune_ResNet34_f42_gpu1 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1
#2.3 恢复训练  通过设置checkpoint选项进行恢复训练：
CUDA_VISIBLE_DEVICES=${cudaid1} python train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" \
    --checkpoint="./fpgm_resnet34_025_120_models/0" >${log_path}/dy_prune_ResNet34_f42_gpu1_load 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1_load

#2.4. 评估  通过调用eval.py脚本，对剪裁和重训练后的模型在测试数据上进行精度：
CUDA_VISIBLE_DEVICES=${cudaid1} python eval.py \
--checkpoint=./fpgm_resnet34_025_120_models/1 \
--model="resnet34" \
--pruned_ratio=0.25 \
--batch_size=128 >${log_path}/dy_prune_ResNet34_f42_gpu1_eval 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1_eval

#2.5. 导出模型   执行以下命令导出用于预测的模型：
CUDA_VISIBLE_DEVICES=${cudaid1} python export_model.py \
--checkpoint=./fpgm_resnet34_025_120_models/final \
--model="resnet34" \
--pruned_ratio=0.25 \
--output_path=./infer_final/resnet > ${log_path}/dy_prune_ResNet34_f42_gpu1_export 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1_export

#add dy_prune_fpgm_mobilenetv1_50_T
CUDA_VISIBLE_DEVICES=${cudaid2} python -m paddle.distributed.launch \
--log_dir="fpgm_mobilenetv1_train_log" \
train.py \
    --model="mobilenet_v1" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.1 \
    --num_epochs=1 \
    --test_period=1 \
    --step_epochs 30 60 90\
    --l2_decay=3e-5 \
    --lr_strategy="piecewise_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_mobilenetv1_models" > ${log_path}/dy_prune_fpgm_mobilenetv1_50_T 2>&1
print_info $? dy_prune_fpgm_mobilenetv1_50_T

#add dy_prune_fpgm_mobilenetv2_50_T
# CUDA_VISIBLE_DEVICES=${cudaid2} python -m paddle.distributed.launch \
# --log_dir="fpgm_mobilenetv2_train_log" \
# train.py \
#     --model="mobilenet_v2" \
#     --data="imagenet" \
#     --pruned_ratio=0.325 \
#     --lr=0.001 \
#     --num_epochs=1 \
#     --test_period=1 \
#     --step_epochs 30 60 80\
#     --l2_decay=1e-4 \
#     --lr_strategy="piecewise_decay" \
#     --criterion="fpgm" \
#     --model_path="./fpgm_mobilenetv2_models" > ${log_path}/dy_prune_fpgm_mobilenetv2_50_T 2>&1
# print_info $? dy_prune_fpgm_mobilenetv2_50_T

#add
CUDA_VISIBLE_DEVICES=${cudaid2} python -m paddle.distributed.launch \
--log_dir="fpgm_resnet34_f_42_train_log" \
train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --batch_size=128 \
    --num_epochs=1 \
    --test_period=1 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" > ${log_path}/dy_prune_ResNet34_f42_gpu2 2>&1
print_info $? dy_prune_ResNet34_f42_gpu2
}

# 3.5 st unstructured_prune
demo_unstructured_prune(){
cd ${slim_dir}/demo/unstructured_prune  || catchException demo_unstructured_prune
# 注意，上述命令中的batch_size为多张卡上总的batch_size，即一张卡的batch_size为256。
## sparsity: -30%, accuracy: 70%/89%
export CUDA_VISIBLE_DEVICES=${cudaid1}
python train.py \
--batch_size 256 \
--pretrained_model ../pretrain/MobileNetV1_pretrained \
--lr 0.05 \
--pruning_mode threshold \
--threshold 0.01 \
--data imagenet \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path st_unstructured_models >${log_path}/st_unstructured_prune_threshold_T 2>&1
print_info $? st_unstructured_prune_threshold_T
# eval
python evaluate.py \
       --pruned_model=st_unstructured_models \
       --data="imagenet"  >${log_path}/st_unstructured_prune_threshold_eval 2>&1
print_info $? st_unstructured_prune_threshold_eval

## sparsity: -55%, accuracy: 67%+/87%+
export CUDA_VISIBLE_DEVICES=${cudaid1}
python train.py \
--batch_size 256 \
--pretrained_model ../pretrain/MobileNetV1_pretrained \
--lr 0.05 \
--pruning_mode ratio \
--ratio 0.55 \
--data imagenet \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path st_ratio_models >${log_path}/st_unstructured_prune_ratio_T 2>&1
print_info $? st_unstructured_prune_ratio_T

# MNIST数据集
# python train.py \
# --batch_size 256 \
# --pretrained_model ../pretrain/MobileNetV1_pretrained \
# --lr 0.05 \
# --pruning_mode threshold \
# --threshold 0.01 \
# --data mnist \
# --lr_strategy piecewise_decay \
# --step_epochs 1 2 3 \
# --num_epochs 1 \
# --test_period 1 \
# --model_period 1 \
# --model_path st_unstructured_models_mnist >${log_path}/st_unstructured_prune_threshold_mnist_T 2>&1
# print_info $? st_unstructured_prune_threshold_mnist_T
# eval
python evaluate.py \
       --pruned_model=st_unstructured_models_mnist \
       --data="mnist"  >${log_path}/st_unstructured_prune_threshold_mnist_eval 2>&1
print_info $? st_unstructured_prune_threshold_mnist_eval

export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
          --log_dir="st_unstructured_prune_gmp_log" \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --model MobileNet \
          --num_epochs 1 \
          --test_period 5 \
          --model_period 10 \
          --pretrained_model ../pretrain/MobileNetV1_pretrained \
          --model_path "./models" \
          --step_epochs  71 88 \
          --initial_ratio 0.15 \
          --pruning_steps 5 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --last_epoch -1 \
          --prune_params_type conv1x1_only \
          --pruning_strategy gmp > ${log_path}/st_unstructured_prune_ratio_gmp 2>&1
print_info $? st_unstructured_prune_ratio_gmp
}
demo_dygraph_unstructured_pruning(){
# dy_threshold
cd ${slim_dir}/demo/dygraph/unstructured_pruning || catchException demo_dygraph_unstructured_pruning
export CUDA_VISIBLE_DEVICES=${cudaid2}
## sparsity: -55%, accuracy: 67%+/87%+
python -m paddle.distributed.launch \
--log_dir train_dy_ratio_log train.py \
--data imagenet \
--lr 0.05 \
--pruning_mode ratio \
--ratio 0.55 \
--batch_size 256 \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path dy_ratio_models >${log_path}/dy_prune_ratio_T 2>&1
print_info $? dy_prune_ratio_T

## sparsity: -30%, accuracy: 70%/89%
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
--log_dir train_dy_threshold_log train.py \
--data imagenet \
--lr 0.05 \
--pruning_mode threshold \
--threshold 0.01 \
--batch_size 256 \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path dy_threshold_models >${log_path}/dy_threshold_prune_T 2>&1
print_info $? dy_threshold_prune_T
# eval
python evaluate.py --pruned_model dy_threshold_models/model.pdparams \
--data imagenet >${log_path}/dy_threshold_prune_eval 2>&1
print_info $? dy_threshold_prune_eval

# load
python -m paddle.distributed.launch \
--log_dir train_dy_threshold_load_log train.py \
--data imagenet \
--lr 0.05 \
--pruning_mode threshold \
--threshold 0.01 \
--batch_size 256 \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 3 \
--test_period 1 \
--model_period 1 \
--model_path dy_threshold_models_new \
--pretrained_model dy_threshold_models/model.pdparams \
--last_epoch 1 > ${log_path}/dy_threshold_prune_T_load 2>&1
print_info $? dy_threshold_prune_T_load
# cifar10
# python train.py --data cifar10 --lr 0.05 \
# --pruning_mode threshold \
# --threshold 0.01 \
# --model_period 1 \
# --num_epochs 2 >${log_path}/dy_threshold_prune_cifar10_T 2>&1
# print_info $? dy_threshold_prune_cifar10_T

export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
          --log_dir="dy_unstructured_prune_gmp_log" \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --num_epochs 1 \
          --test_period 5 \
          --model_period 10 \
          --model_path "./models" \
          --step_epochs 71 88 \
          --initial_ratio 0.15 \
          --pruning_steps 100 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --last_epoch -1 \
          --pruning_strategy gmp \
          --skip_params_type exclude_conv1x1 ${log_path}/dy_unstructured_prune_ratio_gmp 2>&1
print_info $? dy_unstructured_prune_ratio_gmp
}

##################
all_prune(){ # 7个模型
    demo_prune
    demo_dygraph_pruning
    demo_unstructured_prune   # 4个模型
    demo_dygraph_unstructured_pruning
}

#4 nas
demo_nas(){
# 4.1 sa_nas_mobilenetv2
cd ${slim_dir}/demo/nas  || catchException demo_nas
model=demo_nas_sa_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python sa_nas_mobilenetv2.py --search_steps 1 --port 8881 >${log_path}/${model} 2>&1
print_info $? ${model}
}
demo_nas4(){
cd ${slim_dir}/demo/nas || catchException demo_nas4
model=sa_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python sa_nas_mobilenetv2.py --search_steps 1 --retain_epoch 1 --port 8881 >${log_path}/${model} 2>&1
print_info $? ${model}
# 4.2 block_sa_nas_mobilenetv2
model=block_sa_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8883 >${log_path}/${model} 2>&1
print_info $? ${model}

# 4.3 rl_nas
model=rl_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1}  python rl_nas_mobilenetv2.py --search_steps 1 --port 8885 >${log_path}/${model} 2>&1
print_info $? ${model}

# 4.4 parl_nas
#model=parl_nas_v2_T_1card
#CUDA_VISIBLE_DEVICES=${cudaid1} python parl_nas_mobilenetv2.py \
#--search_steps 1 --port 8887 >${log_path}/${model} 2>&1
#print_info $? ${model}
}

all_nas(){ # 3 个模型
   demo_nas
}
# 5 darts
# search 1card # DARTS一阶近似搜索方法
demo_darts(){
cd ${slim_dir}/demo/darts || catchException demo_darts
model=darts1_search_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python search.py --epochs 1 \
--use_multiprocess False \
--batch_size 32 >${log_path}/${model} 2>&1
print_info $? ${model}
#train
model=pcdarts_train_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python train.py --arch='PC_DARTS' \
--epochs 1 --use_multiprocess False \
--batch_size 32 >${log_path}/${model} 2>&1
print_info $? ${model}
# 可视化
#pip install graphviz
#model=slim_darts_visualize_pcdarts
#python visualize.py PC_DARTS > ${log_path}/${model} 2>&1
#print_info $? ${model}
}



slimfacenet(){
cd ${slim_dir}/demo/slimfacenet  || catchException slimfacenet
ln -s ${data_path}/slim/slimfacenet/CASIA CASIA
ln -s ${data_path}/slim/slimfacenet/lfw lfw
model=slim_slimfacenet_B75_train
CUDA_VISIBLE_DEVICES=${cudaid1} python -u train_eval.py \
--train_data_dir=./CASIA/ --test_data_dir=./lfw/ \
--action train --model=SlimFaceNet_B_x0_75 \
--start_epoch 0 --total_epoch 1 >${log_path}/slim_slimfacenet_B75_train 2>&1
print_info $? ${model}
model=slim_slimfacenet_B75_quan
CUDA_VISIBLE_DEVICES=${cudaid1} python train_eval.py \
--action quant --train_data_dir=./CASIA/ \
--test_data_dir=./lfw/  >${log_path}/slim_slimfacenet_B75_quan 2>&1
print_info $? ${model}
model=slim_slimfacenet_B75_eval
CUDA_VISIBLE_DEVICES=${cudaid1} python train_eval.py \
--action test --train_data_dir=./CASIA/ \
--test_data_dir=./lfw/ >${log_path}/slim_slimfacenet_B75_eval 2>&1
print_info $? ${model}
}

all_darts(){  # 2个模型
    demo_darts
    #slimfacenet  需要删掉
}

demo_latency(){
cd ${slim_dir}/demo/analysis  || catchException demo_latency
model=latency_mobilenet_v1_fp32
python latency_predictor.py --model mobilenet_v1 --data_type fp32 >${log_path}/${model} 2>&1
print_info $? ${model}
model=latency_mobilenet_v1_int8
python latency_predictor.py --model mobilenet_v1 --data_type int8 >${log_path}/${model} 2>&1
print_info $? ${model}
model=latency_mobilenet_v2_fp32
python latency_predictor.py --model mobilenet_v2 --data_type fp32 >${log_path}/${model} 2>&1
print_info $? ${model}
model=latency_mobilenet_v2_int8
python latency_predictor.py --model mobilenet_v2 --data_type int8 >${log_path}/${model} 2>&1
print_info $? ${model}
}

all_latency(){
  demo_latency
}

####################################
export all_case_list=(all_distillation all_quant all_prune all_nas  )

export all_case_time=0
declare -A all_P0case_dic
all_case_dic=(["all_distillation"]=5 ["all_quant"]=15 ["all_prune"]=1 ["all_nas"]=30 ["all_darts"]=30 ['unstructured_prune']=15 ['dy_qat1']=1)
for key in $(echo ${!all_case_dic[*]});do
    all_case_time=`expr ${all_case_time} + ${all_case_dic[$key]}`
done
set -e
echo -e "\033[35m ---- P0case_list length: ${#all_case_list[*]}, cases: ${all_case_list[*]} \033[0m"
echo -e "\033[35m ---- P0case_time: $all_case_time min \033[0m"
set +e
####################################
echo -e "\033[35m ---- start run case  \033[0m"
case_num=1
for model in ${all_case_list[*]};do
    echo -e "\033[35m ---- running P0case $case_num/${#all_case_list[*]}: ${model} , task time: ${all_case_list[${model}]} min \033[0m"
    ${model}
    let case_num++
done
echo -e "\033[35m ---- end run case  \033[0m"

cd ${slim_dir}/logs
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    exit 1
else
    exit 0
fi
