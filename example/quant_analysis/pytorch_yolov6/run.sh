export PYTHONPATH=/paddle/xuchang/Quant/PaddleSlim/:$PYTHONPATH
#ps -aux | grep run | awk '{print $2}' | uniq | xargs kill -9
python3.7 run.py --config_path=./config.yaml 
