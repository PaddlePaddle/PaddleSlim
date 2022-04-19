import os
import sys
import paddle
import pickle
from utility import ConvModel, PoolModel, get_logger, model_latency, remove_repeat, latency_metrics, cal_accuracy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

def generate_conv_samples(predictor, Xtest, Ytest, args, logger):
    # array: Xtest, Ytest
    # str: name
    y_pred = predictor.predict(Xtest)

    if not os.path.exists(f'{args.hardware}_samples'):
        os.makedirs(f'{args.hardware}_samples')

    table_dict = {}         
    data_file = f'./{args.hardware}_samples/{args.op_type}_{args.data_type}_aug.pkl'      
    if os.path.exists(data_file):             
        with open(data_file, 'rb') as f:                 
            table_dict = pickle.load(f)

    nums = 0 
    for idx in range(0, len(Xtest)):
        x = Xtest[idx]
        error = abs( (y_pred[idx]-Ytest[idx])/Ytest[idx] )
        if abs(y_pred[idx]-Ytest[idx])<=0.01:
            continue
        if error>0.10:
            nums+=1
            if args.data_type == 'fp32':
                bs, cin, cout, kernel, group, stride, pad, in_hw,_ = x
                in_h = int(np.sqrt(in_hw))
                logger.info(f"idx={nums}, cin={cin}, cout={cout}, k={kernel}, g={group}, s={stride}, p={pad}, in={in_h} latency={Ytest[idx]} pred={y_pred[idx]}")
            else:
                bs, cin, cout, kernel, group, stride, pad, out_hw, flops, params = x
                out_h = int(np.sqrt(out_hw))
                in_h = int((out_h - 1)*stride + kernel - 2*pad)
                logger.info(f"idx={nums}, cin={cin}, cout={cout}, k={kernel}, g={group}, s={stride}, p={pad}, in={in_h}, out={out_h} latency={Ytest[idx]} pred={y_pred[idx]}")

            begin = 0
            end = 0
            step=1
            if int(kernel) == 3:
                begin = max(cout-4, 1)
                end = cout+5
                step=1
            else:
                begin = max(cout-8, 1)
                end = cout+9
                step=2
            if args.data_type == 'fp16':
                begin = max(cout-16, 1)
                end = cout+17
                step=4

            for c_out in range(int(begin), int(end), int(step)):
                if 'depthwise_conv2d' in args.op_type:
                    cin = c_out
                    group = c_out
                else:
                    cin = max(cin, group)
                    c_out = int(round(c_out/group)*group)
                logger.info(f"cout={c_out}, cin={cin}, k={kernel}, s={stride}, p={pad}, g={group}, in_h={in_h}, ")

                paddle.disable_static()
                model = ConvModel(cin=int(cin), cout=int(c_out), kernel=int(kernel), stride=int(stride), group=int(group), pad=int(pad))

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

            if nums%10 == 0:
                with open(data_file, 'wb') as f:         
                    pickle.dump(table_dict, f)
                    print("save successfully")  
    with open(data_file, 'wb') as f:         
        pickle.dump(table_dict, f)
        print("save successfully") 


def generate_pool_samples(predictor, Xtest, Ytest, args, logger):
    # array: Xtest, Ytest
    y_pred = predictor.predict(Xtest)

    if not os.path.exists(f'{args.hardware}_samples'):
        os.makedirs(f'{args.hardware}_samples')

    table_dict = {}         
    data_file = f'./{args.hardware}_samples/{args.op_type}_{args.data_type}_aug.pkl'      
    if os.path.exists(data_file):             
        with open(data_file, 'rb') as f:                 
            table_dict = pickle.load(f)

    nums = 0 
    for idx in range(0, len(Xtest)):
        x = Xtest[idx]
        error = abs( (y_pred[idx]-Ytest[idx])/Ytest[idx] )
        if abs(y_pred[idx]-Ytest[idx])<0.01:
            continue
        if error>0.10:
            nums+=1
            bs, cin, kernel, stride, pad, in_hw, out_hw, flag = x
            pool_type = 'avg' if flag else 'max'
            
            in_h = int(np.sqrt(in_hw))
            out_h = int(np.sqrt(out_hw))
            logger.info(f"idx={nums}, cin={cin}, k={kernel}, s={stride}, p={pad}, in={in_h}, out={out_h}, type={pool_type}, latency={Ytest[idx]} pred={y_pred[idx]}")
            
            begin = int(max(cin-8, 1))
            end = int(cin+9)
            for c_in in range(begin, end, 2):
                logger.info(f"cin={c_in}, k={kernel}, s={stride}, p={pad}, h={in_h}, out_h={out_h}, {pool_type}, ")

                paddle.disable_static()
                adapt_size = 0
                if in_h == kernel:
                    adapt_size = 1
                elif out_h == kernel:
                    adapt_size = out_h
                model = PoolModel(cin=int(c_in), kernel=int(kernel), stride=int(stride), pad=int(pad), adapt_size=int(adapt_size), pool_type=pool_type)

                input_shape = [1, int(c_in), int(in_h), int(in_h)]
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

            if nums%10 == 0:
                with open(data_file, 'wb') as f:         
                    pickle.dump(table_dict, f)
                    print("save successfully")  
    with open(data_file, 'wb') as f:         
        pickle.dump(table_dict, f)
        print("save successfully")        
 

def augment(args):
    logger = get_logger(f'{args.hardware}_{args.op_type}_{args.data_type}_aug')

    model = RandomForestRegressor(
                    max_depth=70,
                    n_estimators=300,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    max_features=0.9,
                    oob_score=True,
                    random_state=10,
                )

    table_dict = {}
    data_file = f'./{args.hardware}_samples/{args.op_type}_{args.data_type}.pkl'
    with open(data_file, 'rb') as f:
        table_dict = pickle.load(f)
    if len(table_dict) == 0:
        logger.info(f"{data_file} is empty!")
        sys.exit()
    acc10 = 0.0 
    times = 0
    target = 0.85
    while acc10 < target:
        if times>=5:
            break
        aug_file = f'./{args.hardware}_samples/{args.op_type}_{args.data_type}_aug.pkl'
        if os.path.exists(aug_file):
            with open(aug_file, 'rb') as f:
                aug_table_dict = pickle.load(f)
            table_dict.update(aug_table_dict)

        data = get_data_from_tables(table_dict, args.op_type, args.data_type)
        X = data[:, 0:-1]
        Y = data[:, -1]
        X, Y = remove_repeat(X,Y)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

        model.fit(Xtrain, Ytrain)
        y_pred = model.predict(Xtest)
        acc10 = cal_accuracy(y_pred, Ytest)
        logger.info(f'acc10: {acc10}')
        if acc10 > target:
            break
        if 'conv2d' in args.op_type:
            generate_conv_samples(model, Xtest, Ytest, args, logger)
        if 'pool2d' in args.op_type:
            generate_pool_samples(model, Xtest, Ytest, args, logger)

        times += 1
    logger.info(f"Data augment is done (times={times}). Final acc10 is {acc10}")

if __name__ == '__main__':
    args = get_args()
    augment(args)