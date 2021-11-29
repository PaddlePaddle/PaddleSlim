import os
import subprocess
import argparse

import paddle
from paddleslim.analysis import TableLatencyPredictor

from paddle.vision.models import mobilenet_v1, mobilenet_v2

opt_tool = 'opt_ubuntu'  # use in linux
# opt_tool = 'opt_M1_mac'     # use in mac with M1 chip
# opt_tool = 'opt_intel_mac'  # use in mac with intel chip

parser = argparse.ArgumentParser(description='latency predictor')
parser.add_argument('--model', type=str, help='which model to test.')
parser.add_argument('--data_type', type=str, default='fp32')

args = parser.parse_args()

if not os.path.exists(opt_tool):
    subprocess.call(
        f'wget https://paddle-slim-models.bj.bcebos.com/LatencyPredictor/{opt_tool}',
        shell=True)
    subprocess.call(f'chmod +x {opt_tool}', shell=True)


def get_latency(model, data_type):
    paddle.disable_static()
    predictor = TableLatencyPredictor(
        f'./{opt_tool}', hardware='845', threads=4, power_mode=3, batchsize=1)
    latency = predictor.predict_latency(
        model,
        input_shape=[1, 3, 224, 224],
        save_dir='./tmp_model',
        data_type=data_type,
        task_type='cls')
    print('{} latency : {}'.format(data_type, latency))

    subprocess.call('rm -rf ./tmp_model', shell=True)
    paddle.disable_static()
    return latency


if __name__ == '__main__':
    if args.model == 'mobilenet_v1':
        model = mobilenet_v1()
    elif args.model == 'mobilenet_v2':
        model = mobilenet_v2()
    else:
        assert False, f'model should be mobilenet_v1 or mobilenet_v2'

    latency = get_latency(model, args.data_type)

    if args.model == 'mobilenet_v1' and args.data_type == 'fp32':
        assert latency == 41.92806607483133
    elif args.model == 'mobilenet_v1' and args.data_type == 'int8':
        assert latency == 36.64814722993898
    elif args.model == 'mobilenet_v2' and args.data_type == 'fp32':
        assert latency == 27.847896889217566
    elif args.model == 'mobilenet_v2' and args.data_type == 'int8':
        assert latency == 23.967800360138803
    else:
        assert False, f'model or data_type wrong.'
