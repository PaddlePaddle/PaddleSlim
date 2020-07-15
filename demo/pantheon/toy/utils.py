import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--out_port",
        type=int,
        default=None,
        help="IP port number for sending out data. (default: %(default)s)")
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="The file path to dump knowledge data. (default: %(default)s)")
    parser.add_argument(
        "--use_cuda",
        type=str2bool,
        default=False,
        help="Whether to use GPU for prediction. (default: %(default)s)")
    parser.add_argument(
        "--test_send_recv",
        type=str2bool,
        default=False,
        help="Whether to test send/recv interfaces. (default: %(default)s)")
    parser.add_argument(
        "--generator_type",
        type=str,
        choices=[
            "sample_generator", "sample_list_generator", "batch_generator"
        ],
        default="batch_generator",
        help="Which data generator to use. (default: %(default)s)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size per device for data generators. (default: %(default)s)"
    )
    parser.add_argument(
        "--serving_times",
        type=int,
        default=1,
        help="The maximum times of teacher serving knowledge. (default: %(default)s)"
    )
    args = parser.parse_args()
    return args


def sample_generator(max_n):
    def wrapper():
        for i in range(max_n):
            yield [i]

    return wrapper


def sample_list_generator(max_n, batch_size=500):
    def wrapper():
        sample_list = []
        for sample in sample_generator(max_n)():
            if len(sample_list) < batch_size:
                sample_list.append(sample)
            if len(sample_list) == batch_size:
                yield sample_list
                sample_list = []
        if len(sample_list) > 0:
            yield sample_list

    return wrapper


# data_generator
def batch_generator(max_n, batch_size=500):
    def wrapper():
        batch = []
        for sample in sample_generator(max_n)():
            if len(batch) < batch_size:
                batch.append(sample)
            if len(batch) == batch_size:
                yield [np.array(batch).astype('int64').reshape((-1, 1))]
                batch = []
        if len(batch) > 0:
            yield [np.array(batch).astype('int64').reshape((-1, 1))]

    return wrapper
