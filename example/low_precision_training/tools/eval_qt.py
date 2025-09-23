# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

# XXX: avoid triggering error on DCU machines
import tarfile

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine_qt, Engine

### quatization
from quantization import Config

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    
    ########## qt
    config.eval_quant=True
    config.upbn=False
    config.qconfigdir = '../qt_config/qt_W4_mse_det.yaml'
    # config.qconfigdir = '../qt_config/qt_W4_minmax_det__A4_mse_det__G4_minmax_sto.yaml'
    # config.qconfigdir = '../qt_config/qt_W4_mse_det__A4_mse_det__G4_minmax_sto.yaml'
    config.qconfig = Config.fromfile(config.qconfigdir)
    ##########

    engine = Engine_qt(config, mode="eval")
    # engine = Engine(config, mode="eval")
    engine.eval()
