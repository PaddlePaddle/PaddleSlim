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

import matplotlib
matplotlib.use('Agg')
import logging
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os

import paddle
import paddle.fluid as fluid

from ..common import get_logger
_logger = get_logger(__name__, level=logging.INFO)


def draw_var_distribution_hist(program,
                               var_names,
                               executor=None,
                               batch_generator=None,
                               data_loader=None,
                               feed_vars=None,
                               fetch_list=None,
                               scope=None,
                               pdf_save_dir='tmp_pdf'):
    """
    Draw hist for distributtion of variables in that name is in var_names
    
    Args:
        program(fluid.Program): program to analyze.
        var_names(list): name of variables to analyze. When there is activation name in var_names, you should set executor, one of batch_generator and data_loader, feed_list.
        executor(fluid.Executor, optional): The executor to run program. Default is None.
        batch_generator(Python Generator, optional):The batch generator provides calibrate data for DataLoader, and it returns a batch every time. For data_loader and batch_generator, only one can be set. Default is None.
        data_loader(fluid.io.DataLoader, optional): The data_loader provides calibrate data to run program. Default is None.
        feed_vars(list): feed variables for program. When you use batch_generator to provide data, you should set feed_vars. Default is None.
        fetch_list(list): fetch list for program. Default is None.
        scope(fluid.Scope, optional): The scope to run program, use it to load variables. If scope is None, will use fluid.global_scope().
        pdf_save_dir(str): dirname to save pdf. Default is  'tmp_pdf'
    
    Returns:
        dict: numpy array of variables that name in var_names
    """
    scope = fluid.global_scope() if scope is None else scope
    assert isinstance(var_names, list), 'var_names is a list of variable name'
    real_names = []
    for var in program.list_vars():
        if var.name in var_names:
            var.persistable = True
            real_names.append(var.name)
    weight_only = False
    if batch_generator is not None:
        assert feed_vars is not None, "When using batch_generator, feed_vars must be set"
        dataloader = fluid.io.DataLoader.from_generator(
            feed_list=feed_vars, capacity=512, iterable=True)
        dataloader.set_batch_generator(batch_generator, executor.place)
    elif data_loader is not None:
        dataloader = data_loader
    else:
        _logger.info(
            "When both batch_generator and data_loader is None, var_names can only include weight names"
        )
        weight_only = True

    if not weight_only:
        assert executor is not None, "when one of batch_generator and data_loader is set, executor must be set"
        assert fetch_list is not None, "when one of batch_generator and data_loader is set, executor must be set"

        for data in dataloader:
            executor.run(program=program,
                         feed=data,
                         fetch_list=fetch_list,
                         return_numpy=False)
            break

    res_np = {}
    for name in real_names:
        var = fluid.global_scope().find_var(name)
        if var is not None:
            res_np[name] = np.array(var.get_tensor())
        else:
            _logger.info(
                "can't find var {}. Maybe you should set one of batch_generator and data_loader".
                format(name))
    numbers = len(real_names)
    if pdf_save_dir is not None:
        if not os.path.exists(pdf_save_dir):
            os.mkdir(pdf_save_dir)
        pdf_path = os.path.join(pdf_save_dir, 'result.pdf')
        with PdfPages(pdf_path) as pdf:
            idx = 1
            for name in res_np.keys():
                if idx % 10 == 0:
                    _logger.info("plt {}/{}".format(idx, numbers))
                arr = res_np[name]
                arr = arr.flatten()
                weights = np.ones_like(arr) / len(arr)
                plt.hist(arr, bins=1000, weights=weights)
                plt.xlabel(name)
                plt.ylabel("frequency")
                plt.title("Hist of variable {}".format(name))
                plt.show()
                pdf.savefig()
                plt.close()
                idx += 1
    return res_np
