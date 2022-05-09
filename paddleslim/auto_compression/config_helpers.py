# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
import yaml
from .strategy_config import *

__all__ = ['save_config', 'load_config']


def load_config(config_path):
    """
        convert yaml to dict config.
    """
    f = open(config_path, 'r')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    compress_config = {}
    for key, value in cfg.items():
        if key == "Global":
            for g_key, g_value in cfg["Global"].items():
                compress_config[g_key] = g_value
        else:
            default_key = eval(key)(**value)
            compress_config[key] = default_key

    if compress_config.get('TrainConfig') != None:
        train_config = compress_config.pop('TrainConfig')
    else:
        train_config = None

    return compress_config, train_config


def save_config(config, config_path):
    """
        convert dict config to yaml.
    """
    f = open(config_path, "w")
    yaml.dump(config, f)
    f.close()
