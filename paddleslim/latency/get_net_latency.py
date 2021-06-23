"""Get model latency from latency lookup table."""
from __future__ import print_function

import re
import argparse
import subprocess


def get_args():
    """Get arguments.

    Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--lookup_table_path',
        default='latency_lookup_table.txt',
        help='Input search space ops path.')
    # parser.add_argument(
    #     '--model_path', default='op_files/MobileNetV2-ops.txt', help='Input model ops path.')

    args = parser.parse_args()
    return args


def get_model_latency(lookup_table_path):
    """Get model latency.
    Args:
        string: search_space_op_path, file path of ops in search space
        string: model_op_path: file path of ops in a model
    Returns:
        float: latency, model latency.
    """
    # fr1 = open(search_space_op_path, 'r')
    # line1 = fr1.readlines()
    # table = dict()
    # for line in line1:
    #     line = line.split()
    #     table[tuple(line[:-1])] = float(line[-1])
    # fr1.close()

    latency = 0.0
    fr2 = open(lookup_table_path, 'r')
    line2 = fr2.readlines()
    n = len(line2)
    for i in range(3, n):
        line = line2[i]
        line = line.split()
        cur_latency = float(line[-3])

        try:
            print("latency:", cur_latency)
            latency += cur_latency
        except:
            print("Current op \" {} \"is not found in look up table...".format(
                ' '.join(line)))

    fr2.close()
    return latency


def main():
    """main."""
    args = get_args()
    total_latency = get_model_latency(args.lookup_table_path)
    print("Input ops path:", args.lookup_table_path)
    print("The total latency of ops in this model is: ", total_latency)


if __name__ == '__main__':
    main()
