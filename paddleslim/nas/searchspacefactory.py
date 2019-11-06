# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from searchspace.registry import SEARCHSPACE

class SearchSpaceFactory(object):
    def __init__(self):
        pass

    def get_search_space(self, key, config):
        """
        get specific model space based on key and config.

        Args:
            key(str): model space name.
            config(dict): basic config information.
        return:
            model space(class)
        """
        cls = SEARCHSPACE.get(key)
        space = cls(config['input_size'], config['output_size'], config['block_num'])

        return space


