import argparse
import pickle


def flops(item):
    return item['flops']


def main(cfgs):
    with open(cfgs.pkl_path, 'rb') as f:
        results = pickle.load(f)

    result.sort(key=flops)

    for item in results:
        assert isinstance(item, dict)
        qualified = True
        if item['flops'] > cfgs.flops:
            qualified = False
        elif 'fid' in item and item['fid'] > cfgs.fid:
            qualified = False
        if qualified:
            print(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pkl_path', type=str, required=True, help='the input .pkl file path')
    parser.add_argument(
        '--flops', type=float, default=5.68e9, help='the FLOPs threshold')
    parser.add_argument(
        '--fid', type=float, default=-1, help='the FID threshold')
    cfgs = parser.parse_args()
    main(cfgs)
