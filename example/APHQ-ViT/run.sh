python test_quant.py \
    --model vit_base --config ./configs/4bit/best.py\
    -cfg ./models/configs_vit/vit_base_patch16_224.yaml \
    -pretrained ./checkpoints/vit_base_patch16_224.pdparams \
    --reconstruct-mlp  --optimize 
    #--load-reconstruct-checkpoint ./checkpoints/ours/vit_small_reconstructed.pth \
    #--load-calibrate-checkpoint ./checkpoints/quant_result/20250608_1544/vit_small_w3_a3_calibsize_128_mse.pth --test-calibrate-checkpoint