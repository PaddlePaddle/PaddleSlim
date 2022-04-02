import os
import sys
sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir)
import argparse

from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create

cfg = load_config('./configs/PaddleDet/ppyoloe_reader.yml')

print(cfg)

coco_loader = create('TestReader')(cfg['TrainDataset'], cfg['worker_num'])

for data in coco_loader:
    print(data.keys())
