export PYTHONPATH=$PWD:$PYTHONPATH

python -m paddle.distributed.launch --log_dir my_log train.py \
    --use_data_parallel=True \
    --batch_size=2048 \
    --lr=1e-3 \
    --l2_decay=3e-5 \
    --total_images=1281167 \
    --class_dim=1000 \
    --image_shape=3,224,224 \
    --model_save_dir=output.ofa.kernel \
    --lr_strategy=piecewise_decay \
    --num_epochs=360 \
    --data_dir=./data/ILSVRC2012 \
    --model=once_for_all_kernel \
    --use_aa=False \
    --checkpoint=./_ofa_epoch359
