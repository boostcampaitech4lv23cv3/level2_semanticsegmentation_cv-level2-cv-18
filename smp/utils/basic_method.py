# # https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import json
import random
import numpy as np
import torch
import torch.nn as nn
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def load_json(path:str)->dict:
    with open(path) as f:
        deployment_def = json.load(f)
    return deployment_def

def fix_seed(seed:int):
    # seed 고정
    print('Seed has been fixed :: {}'.format(seed))
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True # type: ignore    
    torch.backends.cudnn.benchmark = False # type: ignore    
    np.random.seed(random_seed)
    random.seed(random_seed)


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu

