import os
import time
from argparse import ArgumentParser, Namespace
import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *
from models import *
from data_loader import *

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_path', type=str, default='../../input/data')
    parser.add_argument('--anns_file', type=str, default='train_all.json')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    return args

def main(config_path:str, config_file:str, device:str, seed:int,
         data_path:str, anns_file:str,
         batch_size:int, epoch:int, lr:float
        ):
    print(' * Fix Seed')    
    fix_seed(seed)
    print(' * Load Config')
    config = load_json(os.path.join(config_path, config_file))
    print(config)
    train_transform = A.Compose([
                            ToTensorV2()
                            ])
    val_transform = A.Compose([
                              ToTensorV2()
                              ])

    test_transform = A.Compose([
                               ToTensorV2()
                               ])
    # train dataset
    # train_dataset = TrainDataLoader(data_dir=train_path, mode='train', transform=train_transform)

    # validation dataset
    # val_dataset = ValidDataLoader(data_dir=val_path, mode='val', transform=val_transform)

    # test dataset
    # test_dataset = TestDataLoader(data_dir=test_path, mode='test', transform=test_transform)
    # train_loader = DataLoader(dataset=train_dataset, 
                                            #    batch_size=batch_size,
                                            #    shuffle=True,
                                            #    num_workers=4,
                                            #    collate_fn=collate_fn)
     
    # val_loader =  DataLoader(dataset=val_dataset, 
                                            #  batch_size=batch_size,
                                            #  shuffle=False,
                                            #  num_workers=4,
                                            #  collate_fn=collate_fn)
     
    # test_loader = DataLoader(dataset=test_dataset,
                                            #   batch_size=batch_size,
                                            #   num_workers=4,
                                            #   collate_fn=collate_fn)
    print('Start Training')

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)