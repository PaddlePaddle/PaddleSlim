#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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


def compute_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    sub_center = sub_kernel_size // 2
    start = center - sub_center
    end = center + sub_center + 1
    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size):
    assert isinstance(kernel_size, int)
    assert kernel_size % 2 > 0, "kernel size must be odd number"
    return kernel_size // 2


def convert_to_list(value, n):
    return [value, ] * n
