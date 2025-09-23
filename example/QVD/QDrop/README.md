# QDrop
PyTorch implementation of QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization

## Overview

QDrop is a simple yet effective approach, which randomly drops the quantization of activations during reconstruction to pursue flatter model on both calibration and test data. QDrop is easy to implement for various neural networks including CNNs and Transformers, and plug-and-play with little additional computational complexity.

## Integrated into MQBench
Our method has been integrated into quantization benchmark [MQBench](https://github.com/ModelTC/MQBench). And the docs can be found here <https://mqbench.readthedocs.io/en/latest/>. 

**Moreover, obeying the design style of quantization structure in MQBench, we also implement another form of QDrop in branch "qdrop". You can use any branch you like. Details seen in the readme in branch "qdrop"**


## Usage

Go into the exp directory and you can see run.sh and config.sh. run.sh represents a example for resnet18 w2a2. You can run config.sh to produce similar scripts across bits and archs.

run.sh
```
#!/bin/bash
PYTHONPATH=../../../../:$PYTHONPATH \
python ../../../main_imagenet.py --data_path data_path \
--arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2  --act_quant --order together --wwq --waq --awq --aaq \
--weight 0.01 --input_prob 0.5 --prob 0.5
```

config.sh

```
#!/bin/bash
# pretrain models and hyperparameters following BRECQ
arch=('resnet18' 'resnet50' 'mobilenetv2' 'regnetx_600m' 'regnetx_3200m' 'mnasnet')
weight=(0.01 0.01 0.1 0.01 0.01 0.2)
w_bit=(3 2 2 4)
a_bit=(3 4 2 4)
for((i=0;i<6;i++))
do
	for((j=0;j<4;j++))
	do
		path=w${w_bit[j]}a${a_bit[j]}/${arch[i]}
		mkdir -p $path
		echo $path
		cp run.sh $path/run.sh
		sed -re "s/weight([[:space:]]+)0.01/weight ${weight[i]}/" -i $path/run.sh
		sed -re "s/resnet18/${arch[i]}/" -i $path/run.sh
		sed -re "s/n_bits_w([[:space:]]+)2/n_bits_w ${w_bit[j]}/" -i $path/run.sh
		sed -re "s/n_bits_a([[:space:]]+)2/n_bits_a ${a_bit[j]}/" -i $path/run.sh
	done
done
```
Then you can get a series of scripts and run it directly to get the following results.
## Results

Results on low-bit activation in terms of accuracy on ImageNet.

| Methods |  Bits (W/A) | Res18    |Res50 | MNV2 | Reg600M | Reg3.2G | MNasx2 |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   Full Prec. |   32/32 | 71.06 | 77.00 | 72.49 | 73.71 | 78.36 | 76.68   |
|QDrop| 4/4 | 69.07 | 75.03 | 67.91 | 70.81 | 76.36 | 72.81 |
|QDrop| 2/4 | 64.49 | 70.09 | 53.62 | 63.36 | 71.69 | 63.22 |
|QDrop| 3/3 | 65.57 | 71.28 | 55.00 | 64.84 | 71.70 | 64.44 |
|QDrop| 2/2 | 51.76 | 55.36 | 10.21 | 38.35 | 54.00 | 23.62 |



## More experiments

**Case 1, Case 2, Case 3**

To compare the results of 3 Cases mentioned in the observation part of the method, we can use the following commands.

Case 1: weight tuning  doesn't feel any activation quantization

Case 2: weight tuning feels full activation quantization

Case 3: weight tuning feels part activation quantization

```
# Case 1
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order after --wwq --awq --aaq --input_prob 1.0 --prob 1.0
# Case 2
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order before --wwq --waq --aaq --input_prob 1.0 --prob 1.0
# Case 3
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order after --wwq --waq --awq --aaq --input_prob 1.0 --prob 1.0
```

**No Drop**

To compare with QDrop, No Drop can be achieved by turning the probability to 1.0 to disable dropping quantization during weight tuning.

```
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order together --wwq --waq --awq --aaq --input_prob 1.0 --prob 1.0
```

## Reference

If you find this repo useful for your research, please consider citing the paper:

    @article{wei2022qdrop,
	title={QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization},
	author={Wei, Xiuying and Gong, Ruihao and Li, Yuhang and Liu, Xianglong and Yu, Fengwei},
	journal={arXiv preprint arXiv:2203.05740},
	year={2022}
	}