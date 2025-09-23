version=1
num_classes=12


python3 main.py \
    --gpu=1 \
    --seed=3407 \
    --model=BiDfsmn_thinnable_pre  \
    --num_layer=8 \
    --hidden_size=256 \
    --method=no \
    --version=speech_commands_v0.0${version} \
    --num_classes=${num_classes} \
    --test \
    --checkpoint=/mnt/disk1/jsh/2030/paddle/BiFSMN-main/paddle_project/BiDfsmn_thinnable_pre_no_layer8_v1_12.pdparams \
    # --model=BiDfsmn_thinnable_pre --dfsmn_with_bn \
