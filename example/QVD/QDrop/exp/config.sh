#!/bin/bash
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
		# tmux kill-session -t ${arch[i]}_w${w_bit[j]}a${a_bit[j]}
		# cd $path
		# tmux new -s ${arch[i]}_w${w_bit[j]}a${a_bit[j]} -d ./run.sh
		# cd ../..
	done
done
