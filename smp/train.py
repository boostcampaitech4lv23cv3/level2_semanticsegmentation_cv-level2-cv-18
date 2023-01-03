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
from torch.cuda.amp import GradScaler, autocast

from utils import *
from models import *
from data_loader import *

import wandb

class_labels = {
    0: "Background",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--wandb_enable', type=bool, default=True)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--notion_post_enable', type=bool, default=True)
    parser.add_argument('--notion_post_name', type=str, default='{model}')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--config_path', type=str, default='./')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--data_path', type=str, default='../../input/data')
    parser.add_argument('--save_path', type=str, default='../.local/checkpoints')
    parser.add_argument('--save_name', type=str, default='{model}/{time}_best.tar')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--model', type=str, default='FCN_Resnet50')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler_step', type=float, default=25)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--ls', type=float, default=0.0)
    parser.add_argument('--fl_alpha', type=str, default='1.0')
    
    parser.add_argument('--valid_img', type=bool, default=False)
    args = parser.parse_args()
    return args

def init_wandb(args:Namespace):
    if args.wandb_enable :
        wandb.init(
            entity="light-observer",
            project="Trash Segmentation",
            name=args.model if args.wandb_name == None else args.wandb_name,
            config=args.__dict__ # track hyperparameters and run metadata
        )

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
            if args.valid_img == True :
                for i in range(5):
                    # valid_img_setting
                    valid_img = images[i, :, :, :].detach().cpu().numpy()
                    valid_img = np.transpose(valid_img, (1,2,0))
                
                    wandb.log({
                        f'validation_img_{i}': wandb.Image(
                            valid_img,
                            masks={"predictions": {"mask_data": outputs[i, :, :], "class_labels": class_labels},
                                "grund_truth": {"mask_data": masks[i, :, :], "class_labels": class_labels}}   
                        )
                    }, commit=False)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , global_config['Category'])]
        avrg_loss = total_loss / cnt
        
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}') # type: ignore
        print(f'IoU by class : {IoU_by_class}')
        # wandb logging
        if args.wandb_enable :
            log = {
                'val/loss'   :avrg_loss,
                'val/acc'    :acc,
                'val/acc_cls':acc_cls,
                'val/mIoU'   :mIoU,
                'val/fwavacc':fwavacc,
            }
            for kv in IoU_by_class : 
                key = list(kv.keys())[0]
                log['val/IoU_{key}'.format(key=key)] = kv[key]
            wandb.log(log)
        
    return avrg_loss, mIoU

def train(args:Namespace, global_config:dict, model, optimizer, criterion, scheduler, train_loader:DataLoader, val_loader:DataLoader):
    # get current time
    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H.%M.%S')
    output_path = 'No File'

    # get params from args
    n_class = 11
    device = args.device
    num_epochs = args.epoch
    train_best_loss = 9999999
    train_best_mIoU = 0
    val_every = args.val_every
    val_best_loss = 9999999
    val_best_mIoU = 0

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss_array = []
        train_acc_array = []
        train_acc_cls_array = []
        train_mIoU_array = []
        train_fwavacc_array = []
        hist = np.zeros((n_class, n_class))
        with tqdm(total=len(train_loader)) as pbar:
            for step, (images, masks, _) in enumerate(train_loader):
                optimizer.zero_grad()
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                images, masks = images.to(device), masks.to(device) # gpu 연산을 위해 device 할당
                model = model.to(device) # device 할당
                with autocast():
                    # inference
                    outputs = model(images)['out']
                    # loss 계산 (cross entropy loss)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward() # loss.backward()
                scaler.step(optimizer) # optimizer.step()
                scaler.update() # Updates the scale for next iteration.
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

                train_loss_array.append(loss.item())
                train_acc_array.append(acc)
                train_acc_cls_array.append(acc_cls)
                train_mIoU_array.append(mIoU)
                train_fwavacc_array.append(fwavacc)
                pbar.update(1)

        # step schduler
        scheduler.step()

        # Calc train metric
        train_loss = np.average(np.array(train_loss_array))
        train_acc = np.average(np.array(train_acc_array))
        train_acc_cls = np.average(np.array(train_acc_cls_array))
        train_mIoU = np.average(np.array(train_mIoU_array))
        train_fwavacc = np.average(np.array(train_fwavacc_array))
        
        # Print train metric
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {round(train_loss.item(),4)}, mIoU: {round(train_mIoU,4)}')
        if train_best_mIoU < train_mIoU:
            train_best_loss = train_loss
            train_best_mIoU = train_mIoU

        # wandb logging
        if args.wandb_enable :
            log = {
                'train/LR'     : optimizer.param_groups[0]['lr'],
                'train/loss'   : train_loss,
                'train/acc'    : train_acc,
                'train/acc_cls': train_acc_cls,
                'train/mIoU'   : train_mIoU,
                'train/fwavacc': train_fwavacc,
            }
            wandb.log(log)

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_loss, val_mIoU = validation(epoch + 1, model, val_loader, criterion, device, global_config=global_config)
            if val_best_mIoU < val_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                val_best_loss = val_loss
                val_best_mIoU = val_mIoU
                output_path = save_model(best_loss=val_best_loss, best_epoch=epoch + 1, args= args, model=model, optimizer=optimizer, criterion=criterion, time=time)
    
    # post this result to notion
    if(args.notion_post_enable):
        nw = NotionAutoWriter(global_config)
        post_name = args.notion_post_name.format(time=time, model=args.model)
        run = wandb.run
        url = run.get_url()
        content = json.dumps(args.__dict__, indent=4, sort_keys=True)
        nw.post_page(title = post_name, remark = '-', 
                            val_score = round(val_best_mIoU, 4), test_score = 0.0,
                            wandb_link = url, contents = [
                                " * Best Train Score * " , 
                                str(train_best_mIoU) , 
                                " * Best Val Score * " , 
                                str(val_best_mIoU) , 
                                " * Start Time * " , 
                                time , 
                                " * File Path * ",
                                output_path,
                                " * Args * ", 
                                content,
                                ]
                    )


def main(args:Namespace):
    print(' * Fix Seed')    
    fix_seed(args.seed)

    print(' * Load Config')
    global_config = load_json(os.path.join(args.config_path, args.config_file))

    print(' * Create Transforms')
    train_transform = A.Compose([
                                A.RandomResizedCrop(width=args.input_size, height=args.input_size, scale=(0.5, 1.0)),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.5),
                                A.augmentations.transforms.Normalize(),
                                ToTensorV2()
                                ])
    val_transform = A.Compose([
                                A.Resize(args.input_size,args.input_size),
                                A.augmentations.transforms.Normalize(),
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
                                           worker_init_fn=seed_worker, 
                                           drop_last=True,
                                           collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, 
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             worker_init_fn=seed_worker,
                                             collate_fn=collate_fn)

    print(' * Create Model / Criterion / optimizer')
    model = eval('{model}()'.format(model = args.model))
    criterion = get_loss(args=args)
    optimizer = get_optimizer(model, args=args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.scheduler_step, gamma = args.scheduler_gamma)

    print(' * init wandb')
    init_wandb(args=args)

    print(' * Start Training')
    train(args, global_config, model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)