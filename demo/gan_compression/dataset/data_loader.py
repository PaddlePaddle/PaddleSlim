import paddle.fluid as fluid
from data_reader import data_reader


def create_data(cfgs, direction='AtoB', eval_mode=False):
    if eval_mode == False:
        mode = 'TRAIN'
    else:
        mode = 'EVAL'
    reader = data_reader(cfgs, mode=mode)
    data, id2name = reader.make_data(direction)
    loader = fluid.io.DataLoader.from_generator(
        capacity=4, iterable=True, use_double_buffer=True)

    loader.set_batch_generator(
        data,
        places=fluid.CUDAPlace(0)
        if cfgs.use_gpu else fluid.cpu_places())  ### fluid.cuda_places()
    return loader, id2name


def create_eval_data(cfgs, direction='AtoB'):
    return create_data(cfgs, eval_mode=True)
