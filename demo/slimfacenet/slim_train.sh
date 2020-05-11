#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
# ================================================================
#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH='PATH to CUDA and CUDNN'
export FLAGS_fraction_of_gpu_memory_to_use=1.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
SCALE=$1
ARCH=$2
python -u train_eval.py \
    --train_data_dir=/PATH_TO_CASIA_Dataset \
    --test_data_dir=/PATH_TO_LFW \
    --arch=${ARCH} \
    --action final \
    --scale=${SCALE}
