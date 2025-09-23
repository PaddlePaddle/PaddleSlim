n_bits_w=8
n_bits_a=8
run_name="w${n_bits_w}a${n_bits_a}"
sz_ckpt_output_name="sz_w${n_bits_w}a${n_bits_a}"
export CUDA_VISIBLE_DEVICES=0

python3  inference_quant.py \
        --sz_ckpt_path ckpt/sz_w8a8.pth \
        --sz_ckpt_output_path ckpt/${sz_ckpt_output_name}.pth \
        --image TED/ted_test_png/test/_2u_eHHzRto#015410#015606.mp4/0000000.png \
        --motion TED/ted_test_png/motion_32/_2u_eHHzRto#015410#015606.mp4 \
        --cali_image TED/ted_train_png/train/vQILP19qABk#002812#002959.mp4/0000000.png \
        --cali_motion TED/ted_train_png/motion_32/vQILP19qABk#002812#002959.mp4 \
        --steps 32 \
        --n_bits_w $n_bits_w \
        --n_bits_a $n_bits_a \
        --act_quant \
        --save_dir output/pred \
        --resume_sz \
        