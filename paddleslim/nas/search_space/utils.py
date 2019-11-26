import math

def compute_downsample_num(input_size, output_size):
    downsample_num = 0
    while input_size > output_size:
        input_size = math.ceil(float(input_size) / 2.0)
        downsample_num += 1

    if input_size != output_size:
        raise NotImplementedError('output_size must can downsample by input_size!!!')

    return downsample_num
