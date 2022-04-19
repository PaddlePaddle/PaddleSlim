import os
import sys
import paddle
import pickle
from utility import ConvModel, PoolModel, get_logger, model_latency
import numpy as np
import argparse
from paddleslim.analysis.extract_features import get_data_from_tables

def get_args():
    """Get arguments.
        Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('--data_file', default='./conv_fp32.pkl', required=True, help='Dataset')
    parser.add_argument('--op_type', default='conv2d')
    parser.add_argument('--data_type', default='fp32')
    parser.add_argument('--hardware', default=None, required=True, help='Hardware info.')
    parser.add_argument('--backend', default='armv8', help='armv7 or armv8')
    args = parser.parse_args()
    return args

def retest_samples(args):
    logger = get_logger(f'{args.hardware}_{args.op_type}_{args.data_type}')
    op_file = f'./op_samples/{args.op_type}.pkl'
    with open(op_file, 'rb') as f:
        table = pickle.load(f)
    data = get_data_from_tables(table, args.op_type) 

    X = data[:, 0:-1]
    X = X.tolist()
    length = len(X)
    logger.info(f'total samples:{length}')

    if not os.path.exists(f'{args.hardware}_samples'):
        os.makedirs(f'{args.hardware}_samples')
    table_dict = {}         
    data_file = f'./{args.hardware}_samples/{args.op_type}_{args.data_type}.pkl'         
    if os.path.exists(data_file):             
        with open(data_file, 'rb') as f:                 
            table_dict = pickle.load(f)
    
    temp = get_data_from_tables(table_dict, args.op_type, args.data_type)
    start = int(len(temp)-len(temp)%10)
    logger.info(f"start: {start}")
    for idx in range(start, length):
        x = X[idx]
        if 'conv2d' in args.op_type: 
            bs, cin, cout, kernel, group, stride, pad, in_hw, _ = x
            in_h = int(np.sqrt(in_hw))
            
            logger.info(f"idx={idx}, cin={cin}, cout={cout}, k={kernel}, g={group}, s={stride}, p={pad}, in={in_h}x{in_h}")
        
            paddle.disable_static()
            model = ConvModel(cin=int(cin), cout=int(cout), kernel=int(kernel), stride=int(stride), group=int(group), pad=int(pad))

        elif args.op_type == 'pool2d':
            bs, cin, kernel, stride, pad, in_hw, out_hw, flag = x
            pool_type = 'avg' if flag else 'max'
            in_h = int(np.sqrt(in_hw))
            out_h = int(np.sqrt(out_hw))
            adapt_size = 0
            if in_h == kernel:
                adapt_size = 1
            elif out_h == kernel:
                adapt_size = out_h

            logger.info(f"idx={idx}, cin={cin}, k={kernel}, s={stride}, p={pad}, in={in_h}x{in_h}, out={out_h}")

            paddle.disable_static()
            model = PoolModel(cin=int(cin), kernel=int(kernel), stride=int(stride), pad=int(pad), adapt_size=int(adapt_size), pool_type=pool_type)

        input_shape = [1, int(cin), int(in_h), int(in_h)]
        try:
            table_dict = model_latency(model, table_dict, input_shape, args, logger=logger)
        except AssertionError as err:
            logger.info(f"AssertionError: {err}")
            with open(data_file, 'wb') as f:         
                pickle.dump(table_dict, f)
            sys.exit()
        except Exception:
            logger.info("[Test error] or [model error], conitnue.")
            del model
            continue
        del model
        if idx%10 == 0:
            with open(data_file, 'wb') as f:         
                pickle.dump(table_dict, f)
                print("save successfully")  
        
    with open(data_file, 'wb') as f:         
        pickle.dump(table_dict, f)
        print("save successfully") 
    logger.info(f"Done! {args.op_type}_{args.data_type} has been collected")

if __name__ == '__main__':
    args = get_args()
    retest_samples(args)