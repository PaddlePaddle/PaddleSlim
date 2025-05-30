# python main.py ./cifar100_resnet20.hocon 01
python main.py --conf_path ./imagenet_resnet18.hocon --id 02
python main.py --conf_path ./imagenet_resnet18.hocon --id 66 2>&1 | tee w6a6.log