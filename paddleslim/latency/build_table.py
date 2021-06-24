from __future__ import print_function
import sys
import re
import argparse
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))


def get_args():
    """Get arguments.

    Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--op_type',
        default='conv',
        help='Input ops type: conv, fc, batchnorm, pooling, activation.')
    parser.add_argument(
        '--ops_path', default='ops_table.txt', help='ops_table save path.')
    parser.add_argument(
        '--input_dims', default='1,32,224,224', help='dims: NCHW')
    parser.add_argument('--params', \
                         default=None, \
                         help='conv: ch_out, stride, group, kernel, pad, dilation, flag_bias, flag_act \
                               fc: flag_bias, param_dim \
                               bn: epsilon, momentum \
                               pooling: stride, kernel, pad, exclusive, pooling_type \
                               activation: act_type'
                                                    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    handle = open(args.ops_path, 'w')

    input_dims = args.input_dims.replace(',', ' ')

    if args.op_type == 'conv':
        params = args.params.split(',')
        ch_out = params[0]
        stride = params[1]
        groups = params[2]
        kernel = params[3]
        padding = params[4]
        dilation = params[5]
        bias = params[6]
        act = params[7]

        for s in range(1, int(stride) + 1):
            for k in range(1, int(kernel) + 1, 2):
                if k == 1 and s > 1:
                    continue
                for p in range(0, int(padding) + 1):
                    for b in range(0, int(bias) + 1):
                        for a in range(0, int(act) + 1):
                            param_dict = f'ch_out={ch_out}, stride=[{s} {s}], group={groups}, kernel={k}x{k}, pad=[{p} {p} {p} {p}], dilation=[{dilation} {dilation}], flag_bias={b}, flag_act={a}, dtype=float'
                            handle.write('{}\t[{}]\t({})\n'.format( \
                                          args.op_type, input_dims, param_dict.ljust(80)))

    elif args.op_type == 'fc':
        params = args.params.split(',')
        bias = params[0]
        dims = params[1]
        in_ch = input_dims.split(',')[1]

        param_dict = f'flag_bias=0, param_dim={in_ch}x{dims}'
        handle.write('{}\t[{}]\t({})\n'.format( \
                        args.op_type, input_dims, param_dict.ljust(80)))
        param_dict = f'flag_bias=1, param_dim={in_ch}x{dims}'
        handle.write('{}\t[{}]\t({})\n'.format( \
                        args.op_type, input_dims, param_dict.ljust(80)))

    elif args.op_type == 'batchnorm':
        if args.params == None:
            epsilon = 1e-4
            momentum = 0.9
        else:
            params = args.params.split(',')
            epsilon = params[0]
            momentum = params[1]
        param_dict = f'epsilon={epsilon}f, param_dim={momentum}f'
        handle.write('{}\t[{}]\t({})\n'.format( \
                        args.op_type, input_dims, param_dict.ljust(80)))

    elif args.op_type == 'pooling':
        params = args.params.split(',')
        stride = params[0]
        kernel = params[1]
        padding = param[2]
        exclusive = param[3]
        pooling_type = param[4]

        for s in range(1, int(stride) + 1):
            for k in range(1, int(kernel) + 1, 2):
                for p in range(0, int(padding) + 1):
                    for t in ['avg', 'max']:
                        param_dict = f'stride=[{s} {s}], kernel={k}x{k}, pad=[{p} {p} {p} {p}], exclusive={exclusive}, pooling_type={t}'
                        handle.write('{}\t[{}]\t({})\n'.format( \
                        args.op_type, input_dims, param_dict.ljust(80)))

    elif args.op_type == 'activation':
        act_type = args.params
        param_dict = f'act_type={act_type}'

        handle.write('{}\t[{}]\t({})\n'.format( \
                        args.op_type, input_dims, param_dict.ljust(80)))

    handle.close()
    print('Successfully build up ops table!')


if __name__ == '__main__':
    main()
