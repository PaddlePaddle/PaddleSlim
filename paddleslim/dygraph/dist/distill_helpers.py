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

__all__ = ['config2yaml', 'yaml2config']


def yaml2config(yaml_path):
    """
        convert yaml to dict config.
    """
    final_configs = []
    f = open(yaml_path, 'r')
    origin_configs = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    for configs in origin_configs:
        configs = configs['DistillConfig']
        final_configs.extend(configs)
    return final_configs


def config2yaml(configs, yaml_path):
    """
        convert dict config to yaml.
    """
    final_yaml = dict()
    final_yaml['DistillConfig'] = configs
    f = open(yaml_path, "w")
    yaml.dump([final_yaml], f)
    f.close()
