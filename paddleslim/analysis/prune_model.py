import os
import time
import paddle
import paddle.fluid as fluid
from paddleslim.prune import Pruner
from paddleslim.core import GraphWrapper
import numpy as np
__all__ = ["get_sparse_model", "get_prune_model"]


def get_sparse_model(model_file, param_file, ratio):
    assert os.path.exists(model_file) and os.path.exists(
        param_file), f'{model_file} or {param_file} does not exist.'
    paddle.enable_static()

    SKIP = ['image', 'feed', 'pool2d_0.tmp_0']

    folder = os.path.dirname(model_file)
    model_name = model_file.split('/')[-1]
    param_name = param_file.split('/')[-1]

    main_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()
    exe = fluid.Executor(paddle.CPUPlace())
    exe.run(startup_prog)

    [inference_program, feed_target_names, fetch_targets] = (
        fluid.io.load_inference_model(
            folder, exe, model_filename=model_name, params_filename=param_name))
    thresholds = {}

    graph = GraphWrapper(inference_program)
    for op in graph.ops():
        for inp in op.all_inputs():
            name = inp.name()
            if inp.name() in SKIP: continue
            if 'tmp' in inp.name(): continue
            # 1x1_conv
            cond_conv = len(inp._var.shape) == 4 and inp._var.shape[
                2] == 1 and inp._var.shape[3] == 1
            cond_fc = False

            if cond_fc or cond_conv:
                array = np.array(paddle.static.global_scope().find_var(name)
                                 .get_tensor())
                flatten = np.abs(array.flatten())
                index = min(len(flatten) - 1, int(ratio * len(flatten)))
                ind = np.unravel_index(
                    np.argsort(
                        flatten, axis=None), flatten.shape)
                thresholds[name] = ind[0][:index]

    for op in graph.ops():
        for inp in op.all_inputs():
            name = inp.name()
            if name in SKIP: continue
            if 'tmp' in inp.name(): continue

            cond_conv = (len(inp._var.shape) == 4 and inp._var.shape[2] == 1 and
                         inp._var.shape[3] == 1)
            cond_fc = False

            # only support 1x1_conv now
            if not (cond_conv or cond_fc): continue
            array = np.array(paddle.static.global_scope().find_var(name)
                             .get_tensor())
            if thresholds.get(name) is not None:
                np.put(array, thresholds.get(name), 0)
            assert (abs(1 - np.count_nonzero(array) / array.size - ratio) < 1e-2
                    ), 'The model sparsity is abnormal.'
            paddle.static.global_scope().find_var(name).get_tensor().set(
                array, paddle.CPUPlace())

    save_dir = f'./opt_models_tmp/{os.getpid()}_{time.time()}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fluid.io.save_inference_model(
        save_dir,
        feeded_var_names=feed_target_names,
        target_vars=fetch_targets,
        executor=exe,
        main_program=inference_program,
        model_filename=f'sparse_{ratio}.pdmodel',
        params_filename=f'sparse_{ratio}.pdiparams')

    model_file = os.path.join(save_dir, f'sparse_{ratio}.pdmodel')
    param_file = os.path.join(save_dir, f'sparse_{ratio}.pdiparams')

    return model_file, param_file


def get_prune_model(model_file, param_file, ratio):
    assert os.path.exists(model_file) and os.path.exists(
        param_file), f'{model_file} or {param_file} does not exist.'
    paddle.enable_static()

    SKIP = ['image', 'feed', 'pool2d_0.tmp_0']

    folder = os.path.dirname(model_file)
    model_name = model_file.split('/')[-1]
    param_name = param_file.split('/')[-1]

    main_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.global_scope()
    exe.run(startup_prog)

    [inference_program, feed_target_names, fetch_targets] = (
        fluid.io.load_inference_model(
            folder, exe, model_filename=model_name, params_filename=param_name))

    prune_params = []
    graph = GraphWrapper(inference_program)
    for op in graph.ops():
        for inp in op.all_inputs():
            name = inp.name()
            if inp.name() in SKIP: continue
            if 'tmp' in inp.name(): continue
            cond_conv = len(inp._var.shape) == 4 and 'conv' in name
            # only prune conv
            if cond_conv:
                prune_params.append(name)

    # drop last conv
    prune_params.pop()
    ratios = [ratio] * len(prune_params)

    pruner = Pruner()
    main_program, _, _ = pruner.prune(
        inference_program,
        scope,
        params=prune_params,
        ratios=ratios,
        place=place,
        lazy=False,
        only_graph=False,
        param_backup=None,
        param_shape_backup=None)

    save_dir = f'./opt_models_tmp/{os.getpid()}_{time.time()}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fluid.io.save_inference_model(
        save_dir,
        feeded_var_names=feed_target_names,
        target_vars=fetch_targets,
        executor=exe,
        main_program=main_program,
        model_filename=f'prune_{ratio}.pdmodel',
        params_filename=f'prune_{ratio}.pdiparams')

    model_file = os.path.join(save_dir, f'prune_{ratio}.pdmodel')
    param_file = os.path.join(save_dir, f'prune_{ratio}.pdiparams')

    return model_file, param_file
