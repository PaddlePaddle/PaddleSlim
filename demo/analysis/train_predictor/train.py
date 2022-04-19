
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from utility import latency_metrics, remove_repeat, cal_accuracy
from paddleslim.analysis.extract_features import get_data_from_tables
import os
import argparse
def get_args():
    """Get arguments.
        Returns:
        Namespace, arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', default=None, required=True, help='Hardware info.')
    parser.add_argument('--table_file', default='./rk3288_threads_4_power_mode_0.pkl', required=True, help='Dataset')
    parser.add_argument('--op_type', default=None, help='[conv2d, depthwise_conv2d, pool2d...]')
    parser.add_argument('--data_type', default='fp32', help='fp32, int8')
    args = parser.parse_args()
    return args

def train_predictor(args):
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
    with open(args.table_file, 'rb') as f:
        table_dict = pickle.load(f)

    op_file = f'./{args.hardware}_samples/{args.op_type}_{args.data_type}.pkl' 
    if os.path.exists(op_file):
        with open(op_file, 'rb') as f:
            new_table_dict = pickle.load(f)
        table_dict.update(new_table_dict)

    aug_file = f'./{args.hardware}_samples/{args.op_type}_{args.data_type}_aug.pkl'
    if os.path.exists(aug_file):
        with open(aug_file, 'rb') as f:
            new_table_dict = pickle.load(f)
        table_dict.update(new_table_dict)
    
    data = get_data_from_tables(table_dict, args.op_type, args.data_type)
    print("samples:", len(data))
    X = data[:, 0:-1]
    Y = data[:, -1]
    X, Y = remove_repeat(X,Y)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
    model_rf = model.fit(X, Y)

    y_pred = model_rf.predict(Xtest)
    acc10 = cal_accuracy(y_pred, Ytest)
    
    save_path = args.table_file.split('.')[0] + '_batchsize_1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if 'conv2d' in args.op_type:
        path = f'{save_path}/{args.op_type}_{args.data_type}_predictor.pkl'
    elif 'matmul' in args.op_type:
        path = f'{save_path}/matmul_predictor.pkl'
    else:
        path = f'{save_path}/{args.op_type}_predictor.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_rf, f)
        print(path)
        print("save successfully!\n")
        

    with open(args.table_file, 'wb') as f:         
        pickle.dump(table_dict, f)

if __name__ == '__main__':
    args = get_args()
    if args.op_type:
        train_predictor(args)
    else:
        for op_type in ['depthwise_conv2d', 'conv2d', 'pool2d', 'matmul_v2', 'elementwise_add', 'elementwise_mul', 'concat', 'calib', 'swish']:
            if args.data_type == 'fp32' and op_type == 'calib':
                continue
            args.op_type = op_type
            train_predictor(args)
           
   