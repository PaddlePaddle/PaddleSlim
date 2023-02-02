export FLAGS_cudnn_deterministic=true
export CUDA_VISIBLE_DEVICES=2
export FLAGS_fuse_parameter_memory_size=32
export FLAGS_fuse_parameter_groups_size=50

export CUDA_VISIBLE_DEVICES=2,3,4,6
timeout 10m python distill.py --batch_size 256 --fuse_reduce > log_pe_card4_fuse 2>&1
ps -ef | grep distill | awk '{print $2}' | xargs kill -9
timeout 10m python -m paddle.distributed.launch distill.py --fleet --batch_size 64 --fuse_reduce > log_fleet_card4_fuse 2>&1
ps -ef | grep distill | awk '{print $2}' | xargs kill -9