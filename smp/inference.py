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
    parser.add_argument('--config_path', type=str, default='./')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--data_path', type=str, default='../../input/data')
    parser.add_argument('--sample_path', type=str, default='../submission')
    parser.add_argument('--sample_name', type=str, default='sample_submission.csv')
    parser.add_argument('--submission_path', type=str, default='../.local/submission')
    parser.add_argument('--submission_name', type=str, default='{time}_{model}_submission.csv')
    parser.add_argument('--save_path', type=str, default='../.local/checkpoints')
    parser.add_argument('--save_file', type=str)
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    return args

def test(args:Namespace, model, test_loader):
    INFERENCE_SIZE = 256 # This is a constant
    transform = A.Compose([A.Resize(INFERENCE_SIZE, INFERENCE_SIZE)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, INFERENCE_SIZE*INFERENCE_SIZE), dtype=np.int32)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(args.device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], INFERENCE_SIZE*INFERENCE_SIZE]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def submission(args:Namespace, model, test_loader):
    # make dir
    os.makedirs(args.submission_path, exist_ok=True)

    # get current time
    now = datetime.now()
    time = now.strftime('%Y-%m-%d %H:%M:%S')

    # set path
    sample_path = os.path.join(args.sample_path, args.sample_name)
    submission_path = os.path.join(args.submission_path, args.submission_name.format(time=time, model=model.__class__.__name__))

    # sample_submisson.csv 열기
    submission = pd.read_csv(sample_path, index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, args.device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                       ignore_index=True)

    # submission.csv로 저장

    submission.to_csv(submission_path, index=False)

def main(args:Namespace):

    print(' * Load Config')
    global_config = load_json(os.path.join(args.config_path, args.config_file))
    print(global_config)

    print(' * Create Transforms')
    test_transform = A.Compose([
                               ToTensorV2()
                               ])

    print(' * Create Datasets')
    test_dataset = BaseDataset(dataset_path=args.data_path, mode='test', transform=test_transform, global_config = global_config)

    print(' * Create Loaders')
    test_loader = DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    print(' * Create Model / Criterion / optimizer')
    model = FCN_Resnet50()

    print(' * Start Inference')


if __name__ == '__main__':
    args = parse_args()
    main(args)