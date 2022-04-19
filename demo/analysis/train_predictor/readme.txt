准备文件：将已打好的延时表 rk3288_threads_4_power_mode_0_batchsize_1 放在auto目录下。

1.采集op数据，需要采集conv2d(fp32\int8), depthwise_conv2d(fp32\int8), pool2d，执行指令如下：
nohup python sample_data.py --hardware rk3288 --op_type conv2d --data_type fp32 --backend armv7  1>/dev/null 2>error.log &
nohup python sample_data.py --hardware rk3288 --op_type depthwise_conv2d --data_type fp32 --backend armv7  1>/dev/null 2>error.log  &
nohup python sample_data.py --hardware rk3288 --op_type pool2d --data_type fp32 --backend armv7  1>/dev/null 2>error.log  &

可在 log/{args.hardware}_{args.op_type}_{args.data_type}.log 文件中查看op数据采集进度。如果程序意外中断了，再次执行上述指令即可，程序会继续采集。

2. 基于op数据，测试op预测器的准确率，决定是否进行增强训练，执行指令如下：
nohup python augment.py --hardware rk3288 --op_type conv2d --data_type fp32 --backend armv7 1>/dev/null 2>error.log &
nohup python augment.py --hardware rk3288 --op_type depthwise_conv2d --data_type fp32 --backend armv7 1>/dev/null 2>error.log & 
nohup python augment.py --hardware rk3288 --op_type pool2d --data_type fp32 --backend armv7 1>/dev/null 2>error.log &

可在 log/{args.hardware}_{args.op_type}_{args.data_type}_aug.log 文件中查看op数据采集进度。如果程序意外中断了，再次执行上述指令即可，程序会继续采集。

3. 训练并保存op预测器，执行指令如下：
nohup python train.py  --hardware rk3288 --table_file rk3288_threads_4_power_mode_0.pkl 1>/dev/null 2>error.log &

op_type预测器将会保存在 rk3288_threads_4_power_mode_0_batchsize_1 文件夹下。使用预测器时，将该 rk3288_threads_4_power_mode_0_batchsize_1 文件夹 和 rk3288_threads_4_power_mode_0.pkl

ps: 第一步和第二步都需要连接设备，第三步不需要。