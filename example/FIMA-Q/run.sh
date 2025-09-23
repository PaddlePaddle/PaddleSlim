python test_quant.py \
    --model vit_small --config ./configs/4bit/best.py \
    -cfg ./models/configs_vit/vit_small_patch16_224.yaml \
    -pretrained ./checkpoints/vit_small_patch16_224.pdparams \
    --optimize --optim-metric fisher_dplr 
  #--load-calibrate-checkpoint ./checkpoints/quant_result/20250608_1544/vit_small_w3_a3_calibsize_128_mse.pth --test-calibrate-checkpoint