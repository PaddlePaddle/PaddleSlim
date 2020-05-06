# ================================================================
#   Copyright (C) 2020 BAIDU CORPORATION. All rights reserved.
#   
#   Filename   :    slim_eval.sh
#   Author     :    paddleslim@baidu.com
#   Date       :    2020-05-06
#   Describe   :    eval the performace of slimfacenet on lfw
#
# ================================================================

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export LD_LIBRARY_PATH='PATH to CUDA and CUDNN'
python eval_infer_model.py --action test
