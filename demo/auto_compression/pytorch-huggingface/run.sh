export PYTHONPATH=/paddle/auto_compress/final_git/PaddleSlim/:$PYTHONPATH
export PYTHONPATH=/paddle/auto_compress/PaddleNLP/:$PYTHONPATH

python3.7 run.py --config_path=./configs/cola.yaml --save_dir='./output/cola/'
