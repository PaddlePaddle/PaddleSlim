# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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
import os

__all__ = ['load_config', 'save_config']


def print_arguments(args, level=0):
    if level == 0:
        print('-----------  Running Arguments -----------')
    for arg, value in sorted(args.items()):
        if isinstance(value, dict):
            print('\t' * level, '%s:' % arg)
            print_arguments(value, level + 1)
        else:
            print('\t' * level, '%s: %s' % (arg, value))
    if level == 0:
        print('------------------------------------------')


def load_config(config):
    """Load configurations from yaml file into dict.
    Fields validation is skipped for loading some custom information.
    Args:
      config(str): The path of configuration file.
    Returns:
      dict: A dict storing configuration information.
    """
    if config is None:
        return None
    assert isinstance(
        config,
        str), f"config should be str but got type(config)={type(config)}"
    assert os.path.exists(config) and os.path.isfile(
        config), f"{config} not found or it is not a file."
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print_arguments(cfg)
    return cfg


def save_config(config, config_path):
    """
        convert dict config to yaml.
    """
    f = open(config_path, "w")
    yaml.dump(config, f)
    f.close()
