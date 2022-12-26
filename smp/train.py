import os
from datetime import datetime
from argparse import ArgumentParser, Namespace
import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torchvision import models

from utils import *
from models import *
from data_loader import *

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--config_path', type=str, default='./')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--data_path', type=str, default='../../input/data')
    parser.add_argument('--save_path', type=str, default='../.local/checkpoints')
    parser.add_argument('--save_name', type=str, default='{model}/{time}_best.tar')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--model', type=str, default='FCN_Resnet50')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    return args

def validation(epoch:int, model, data_loader:DataLoader, criterion, device:str, global_config:dict):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , global_config['Category'])]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}') # type: ignore
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss

def train(args:Namespace, global_config:dict, model, optimizer, criterion, train_loader, val_loader):
    # get current time
    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H.%M.%S')

    # get params from args
    n_class = 11
    best_loss = 9999999
    num_epochs = args.epoch
    device = args.device
    val_every = args.val_every

    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)['out']
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device, global_config=global_config)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                best_loss = avrg_loss
                save_model(best_loss=best_loss, best_epoch=epoch + 1, args= args, model=model, optimizer=optimizer, criterion=criterion, time=time)

def main(args:Namespace):
    print(' * Fix Seed')    
    fix_seed(args.seed)

    print(' * Load Config')
    global_config = load_json(os.path.join(args.config_path, args.config_file))

    print(' * Create Transforms')
    train_transform = A.Compose([
                            ToTensorV2()
                            ])
    val_transform = A.Compose([
                              ToTensorV2()
                              ])
                              
    print(' * Create Datasets')
    train_dataset = BaseDataset(dataset_path=args.data_path, mode='train', transform=train_transform, global_config = global_config)
    val_dataset = BaseDataset(dataset_path=args.data_path, mode='val', transform=val_transform, global_config = global_config)

    print(' * Create Loaders')
    train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, 
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=collate_fn)

    print(' * Create Model / Criterion / optimizer')
    model = eval('{model}()'.format(model = args.model))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    print(' * Start Training')
    train(args, global_config, model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader, val_loader=val_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)