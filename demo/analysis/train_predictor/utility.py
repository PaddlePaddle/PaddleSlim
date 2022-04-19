import re
import os
import math 
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
import subprocess
import paddle
from paddle.nn import Conv2D, BatchNorm2D, ReLU, MaxPool2D, AvgPool2D, AdaptiveAvgPool2D, AdaptiveMaxPool2D
import pickle
import paddleslim
import logging
from paddle.static import InputSpec
from paddleslim.analysis.parse_ops import get_key_from_op

def get_accuracy(y_pred, y_true, threshold=0.1):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)

def cal_accuracy(y_pred, y_true, threshold1=0.1, threshold2=0.01):     
    a = (y_true - y_pred) / y_true     
    b = abs(y_true - y_pred) 
    res1 = np.where(abs(a) <= threshold1) 
    res2 = np.where(abs(b) <= threshold2)
    res = list(res1[0]) + list(res2[0])
    res = tuple(set(res))  
    return len(res) / len(y_true)

def get_maxError(y_pred, y_true):
    a = abs((y_true - y_pred) / y_true)
    return max(a)

def latency_metrics(y_pred, y_true):
    """
    evaluation metrics for prediction performance
    """
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    acc20 = get_accuracy(y_pred, y_true, threshold=0.20)
    max_error = get_maxError(y_pred, y_true)
    new_acc10 = cal_accuracy(y_pred, y_true, 0.1, 0.01)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15, acc20, max_error, new_acc10

def cal_flops_params(op_type, cin, cout, kernel=1, h=1, w=1):
    # cin: weight[1]
    if 'conv' in op_type:
        params = cout * (kernel * kernel * cin+ 1)
        flops = 2 * kernel * kernel * h * w * cin * cout
        return flops, params
    elif "fc" in op_type:
        flops = 2 * cin * cout
        params = (cin + 1) * cout
        return flops, params

    
def model_latency(model, table_dict, input_shape, args, save_dir='./model', logger=None):
    # input_shape: list
    model_file, param_file = save_cls_model(model, input_shape, save_dir, args.data_type)

    opt_dir = './tools/opt_mac_intel_v2_10'

    enable_fp16 = True if args.data_type=='fp16' else False
    
    pb_model = opt_model(opt=opt_dir, model_file = model_file, param_file = param_file,  optimize_out_type='protobuf', enable_fp16=enable_fp16)

    paddle.enable_static()
    with open(pb_model, "rb") as f:
        fluid_program = paddle.fluid.framework.Program.parse_from_string(f.read())
    graph = paddleslim.core.GraphWrapper(fluid_program)
    run_test = False
    # for op in graph.ops():  # if all op are already in table_dict, skip test.
    #     param_key = get_key_from_op(op, data_type=args.data_type)
    #     # print(param_key)
    #     if param_key != '' and param_key not in table_dict:
    #         run_test = True
    #         break
    
    # if not run_test:
    #     print('This model has been tested before. Now skip the test.')
    #     # time.sleep(1)
    #     return table_dict
    nb_model = opt_model(opt=opt_dir, model_file = model_file, param_file = param_file,  optimize_out_type='naive_buffer', enable_fp16=enable_fp16)

    inputs = ','.join(str(i) for i in input_shape)
    latency_info, latency = test_on_device(nb_model, inputs, 4, 0, backend=args.backend)
    table_dict = update_table(args, graph, latency_info, table_dict, logger) 

    del graph     
    del fluid_program
    return  table_dict  


def opt_model(opt="./tools/opt_mac_intel_v2_10",
              model_file='',
              param_file='',
              optimize_out_type='protobuf',
              valid_targets='arm',
              enable_fp16=False):
    assert os.path.exists(opt), f"{opt} doesn't exist!"
    assert os.path.exists(model_file) and os.path.exists(
        param_file), f'{model_file} or {param_file} does not exist.'
    save_dir = f'./opt_models_tmp/{os.getpid()}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert optimize_out_type in ['protobuf', 'naive_buffer']
    if optimize_out_type == 'protobuf':
        model_out = os.path.join(save_dir, 'pbmodel')
    else:
        model_out = os.path.join(save_dir, 'model')
    enable_fp16 = str(enable_fp16).lower()
    cmd = f'{opt} --model_file={model_file} --param_file={param_file}  --optimize_out_type={optimize_out_type} --optimize_out={model_out} --valid_targets={valid_targets} --enable_fp16={enable_fp16}'
    print(f'commands:{cmd}')
    m = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = m.communicate()
    print(out, 'opt done!')

    if optimize_out_type == 'protobuf':
        model_out = os.path.join(model_out, 'model')
    else:
        model_out = model_out + '.nb'
    return model_out

def test_on_device(nb_model, input_shape, threads, power_mode, tool_dir='./tools', use_detect_system=False, backend='armv8'):
    check_dev_connect()
    assert os.path.exists(nb_model), f"{nb_model} doesn't exist!"
    cmd = f'adb push {nb_model} /data/local/tmp'
    m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = m.communicate()
    
    benchmark_bin = f'benchmark_bin_{backend}_profile'
    # 
    assert os.path.exists(os.path.join(tool_dir, benchmark_bin)), f"{os.path.join(tool_dir, benchmark_bin)} doesn't exist!"
    
    
    if use_detect_system:
        modelpath = os.path.dirname(nb_model)
        cmd = f'adb push {tool_dir}/config.txt /data/local/tmp && adb push {tool_dir}/test.jpg /data/local/tmp && adb push {tool_dir}/coco_label_list.txt /data/local/tmp '
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out = m.communicate()

        cmd = f'adb push {tool_dir}/libpaddle_light_api_shared.so /data/local/tmp && adb push {tool_dir}/detect_system_profile /data/local/tmp'
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out = m.communicate()

        cmd = 'adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH && chmod +x detect_system_profile && ./detect_system_profile config.txt test.jpg 1>profile.log 2>&1" '   
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out = m.communicate()

    else:
        cmd = f'adb push {os.path.join(tool_dir, benchmark_bin)} /data/local/tmp'
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out = m.communicate()

        if 'v2_9' in benchmark_bin:
            cmd = f'adb shell "cd /data/local/tmp && ./{benchmark_bin} --input_shape={input_shape} --optimized_model_path=model.nb --warmup=100 --repeats=100 --threads={threads} --power_mode={power_mode} 1>profile.log 2>&1" '
        else:
            # v2.10
            cmd = f'adb shell "cd /data/local/tmp && ./{benchmark_bin} --input_shape={input_shape} --optimized_model_file=model.nb --warmup=100 --repeats=100 --threads={threads} --power_mode={power_mode} --backend=arm 1>profile.log 2>&1" '
        m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out = m.communicate()
    
    cmd = 'adb pull /data/local/tmp/profile.log  .'
    m = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = m.communicate()
    print("profile done!")
    
    latency_info, total_latency = profile_log('./profile.log')
    print('avg_len:', len(latency_info)-1)
    if total_latency <= 0:
        raise Exception('test failed!')   
       
    return latency_info, total_latency


def update_table(args, graph, latency_info, table_dict, logger=None):
    latency = latency_info[1:]
    is_inorder = True
    all_param_keys = []
    idx = 0
    for op in graph.ops():
        if op.type() in ['feed', 'fetch']:
            continue
        param_key = get_key_from_op(op)
        if param_key != '':
            all_param_keys.append(param_key)
            nb_type, nb_input, nb_weight, nb_output, nb_time = latency[idx].split('\t')
            if param_key.split()[0] != nb_type:
                print(f'The op doesn\'t match.\nidx:{idx}\t{param_key.split()[0]}!={nb_type}. is_inorder is False')
                is_inorder = False
            idx += 1
    
    if is_inorder:
        for idx, param_key in enumerate(all_param_keys):
            nb_time = float(latency[idx].split('\t')[-1])
            
            flag = 0
            if param_key in table_dict:
                flag = 1
                nb_time = (nb_time+table_dict[param_key])/2.0
            
            if logger and args.op_type in param_key:
                logger.info(f"{flag}_latency={nb_time}")
            table_dict.update({param_key:nb_time})

    else:
        for idx, param_key in enumerate(all_param_keys):
            flag = 0
            found = False
            cur_idx = idx
            for i in range(0, 2*len(latency)):
                if i%2==0:
                    cur_idx -= i
                else:
                    cur_idx += i
                if cur_idx < 0 or cur_idx >= len(latency):
                    continue

                profile_info = latency[cur_idx]
                nb_type, nb_input, nb_weight, nb_output, nb_time = profile_info.split('\t')

                if param_key.split()[0] != nb_type:
                    continue

                if '-1,' not in param_key and 'scale' not in param_key:
                    if nb_input != 'N/A': 
                        if 'in=' in param_key:
                            if '/' in nb_input:
                                nb_input = ''.join('(' + ', '.join(s.split('x')) + ')' for s in nb_input.split('/'))
                            else:
                                if len(nb_input.split('x'))==1:
                                    nb_input = '(' + nb_input + ',)'
                                else:
                                    nb_input = '(' + ', '.join(nb_input.split('x')) + ')'
                                if 'in=()' in param_key:
                                    nb_input = 'in=()'
                            
                        # TODO
                        elif ('X' in nb_input and 'Y' in nb_input) and ('X=' in param_key and 'Y=' in param_key):
                            x_info = nb_input.split('X')[-1].split('Y')[0]
                            y_info = nb_input.split('Y')[-1]
                            if nb_type=='greater_equal':
                                x_info = x_info.split(':')[1]
                                y_info = y_info.split(':')[1]
                            if len(x_info.split('x'))==1:
                                nb_input = 'X=(' + x_info + ',) '
                            else:
                                nb_input = 'X=(' + ', '.join(x_info.split('x')) + ') '
                            
                            if 'X=()' in param_key:
                                nb_input = 'X=() '
                            
                            if len(y_info.split('x'))==1:
                                nb_input += 'Y=(' + y_info + ',)'
                            else:
                                nb_input += 'Y=(' + ', '.join(y_info.split('x'))+')'
                            if 'Y=()' in param_key:
                                nb_input = nb_input.split('Y')[0]
                                nb_input += 'Y=()'
                        else:
                            nb_input = nb_type

                        if nb_input not in param_key and nb_type!='fc':
                            continue
                    
                    if nb_weight != 'N/A' and 'weight=' in param_key:
                        nb_weight = 'weight=(' + ', '.join(nb_weight.split('x')) + ')'
                        if nb_weight not in param_key:
                            continue

                    if nb_output != 'N/A' and 'out=' in param_key:
                        if len(nb_output.split('x'))==1:
                            nb_output = 'out=(' + nb_output + ',)'
                        else:
                            nb_output = 'out=(' + ', '.join(nb_output.split('x')) + ')'
                        if nb_output not in param_key:
                            continue
                
                if param_key in table_dict:
                    flag = 1
                    nb_time = (nb_time+table_dict[param_key])/2.0
            
                if logger and args.op_type in param_key:
                    logger.info(f"{flag}_latency={nb_time}")
                table_dict.update({param_key:nb_time})

                found = True 
                break 
            assert found == True, f'can not find the corresponding key to {param_key}'

    return table_dict


class ConvModel(paddle.nn.Layer):
    def __init__(self, cin, cout, kernel, stride, group=1, pad=0):
        super(ConvModel, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1
        self.conv = Conv2D(in_channels=cin, out_channels=cout, kernel_size=kernel, stride=stride, padding=pad, dilation=1, groups=group)
        # input channels
        self.bn = BatchNorm2D(cout)
        self.relu = ReLU()
        # self.pool = AvgPool2D(kernel_size=kernel,stride=stride, padding=pad)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        out = self.relu(x)
        # out = self.pool(out)
        return out

# 必须有bn，不然模型中没参数变量，保存模型会报错
class PoolModel(paddle.nn.Layer):
    def __init__(self, cin, kernel, stride, pad=0, adapt_size=0, pool_type='avg'):
        super(PoolModel, self).__init__()
        self.pool = None
        if pool_type == 'avg':
            self.pool = AvgPool2D(kernel_size=kernel,stride=stride, padding=pad)
            if adapt_size != 0:
                self.pool = AdaptiveAvgPool2D(adapt_size)
        else:
            self.pool = MaxPool2D(kernel_size=kernel,stride=stride, padding=pad)
            if adapt_size != 0:
                self.pool = AdaptiveMaxPool2D(adapt_size)
        self.bn = BatchNorm2D(cin)
    def forward(self, inputs):
        out = self.pool(inputs)
        out = self.bn(out)
        return out


def save_cls_model(model, input_shape, save_dir='./model', data_type='fp32'):
    def sample_generator(input_shape, batch_num):
        def __reader__():
            for i in range(batch_num):
                image = np.random.random(input_shape).astype('float32')
                yield image
        return __reader__
    static_model = paddle.jit.to_static(model, input_spec=[InputSpec(shape=input_shape, dtype='float32', name='inputs')])
    paddle.jit.save(
        static_model,
        path=os.path.join(save_dir, 'fp32model'))
    model_file = os.path.join(save_dir, 'fp32model.pdmodel')
    param_file = os.path.join(save_dir, 'fp32model.pdiparams')
    
    if data_type == 'int8':
        save_dir = os.path.dirname(model_file)
        quantize_model_path = os.path.join(save_dir, 'int8model')
        if not os.path.exists(quantize_model_path):
            os.makedirs(quantize_model_path)
        
        paddle.enable_static()
        exe = paddle.fluid.Executor(paddle.fluid.CPUPlace())
        
        paddleslim.quant.quant_post_static(
            executor=exe,
            model_dir=save_dir,
            quantize_model_path=quantize_model_path,
            sample_generator=sample_generator(input_shape, input_shape[0]),
            model_filename=model_file.split('/')[-1],
            params_filename=param_file.split('/')[-1],
            batch_size=input_shape[0],
            batch_nums=input_shape[0],
            weight_bits=8,
            activation_bits=8)
        model_file = os.path.join(quantize_model_path, '__model__')
        param_file = os.path.join(quantize_model_path, '__params__')
    paddle.jit.clear_function_cache()
    return model_file, param_file


def remove_repeat(Xtrain, Ytrain):
    # array: Xtrain, Ytrain
    d = {}
    # print("before remove:", len(Xtrain))
    Xtrain = Xtrain.tolist()

    for idx in range(len(Xtrain)):
        x = Xtrain[idx]
        y = Ytrain[idx]
        if str(x) in d:
                error = abs(d[str(x)]-y)/(max(d[str(x)], y)+0.001)
                if error<0.1 or abs(d[str(x)]-y)<=0.01:
                    y = (y + d[str(x)])/2.0
                else:
                    y = min(d[str(x)], y)
        d.update({str(x):y})
    import json
    x = []
    y = []
    for key in d:
        x.append(json.loads(key))
        y.append(d[key])
    # print("after remove:", len(x))
    return np.array(x), np.array(y)

def check_dev_connect():
    cmd = 'adb devices | grep device'
    dev_info = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = dev_info.communicate()[0]
    res = out.decode().find("\tdevice")
    assert res != -1, "No device is attached"

def profile_log(profile_log_path):
    fr = open(profile_log_path, 'r')
    lines = fr.readlines()
    lines = lines[::-1]
    target = "Detailed Dispatch Profiler Summary"
    target_line = 0
    for i in range(len(lines)):
        if lines[i].find(target) != -1:
            target_line = i
            break

    print('found:-', target_line)
    txtName = "latency.txt"
    latency_info = []
    f = open(txtName, "w+")
    while(target_line != 0):
        if lines[target_line].find("Summary information") != -1:
            break

        temp = lines[target_line].split()
        if len(temp) == 14:
            new_context = temp[0] + '\t' + temp[2] + '\t' + temp[3] + '\t' + temp[4] + '\t' + temp[5] + '\t'+ temp[6]+ '\t' +temp[7]+'\n'
            f.write(new_context)
            latency_info.append(temp[0] + '\t' + temp[4] + '\t' + temp[5] + '\t'+ temp[6]+ '\t' + temp[7])
        target_line -= 1
    fr.close()
    f.close()
    
    total_latency = 0.0
    for info in latency_info[1:]:
        total_latency += float(info.split('\t')[-1])

    print('total_latency = {}ms'.format(total_latency))
    return latency_info, total_latency


def get_logger(name, level=logging.INFO, fmt='%(message)s '):
    """
    Get logger from logging with given name, level and format without
    setting logging basicConfig. For setting basicConfig in paddle
    will disable basicConfig setting after import paddle.
    Args:
        name (str): The logger name.
        level (logging.LEVEL): The base level of the logger
        fmt (str): Format of logger output
    Returns:
        logging.Logger: logging logger with given setttings
    Examples:
    .. code-block:: python
       logger = log_helper.get_logger(__name__, logging.INFO,
                            fmt='%(asctime)s-%(levelname)s: %(message)s')
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # handler = logging.StreamHandler() # sys.stderr
    if not os.path.exists('./log'):
        os.makedirs('./log')
    handler = logging.FileHandler(f'log/{name}.log')
    if fmt:
        formatter = logging.Formatter(fmt=fmt)
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = 0
    return logger