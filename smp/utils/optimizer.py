import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn

def get_optimizer(model, args:Namespace) -> torch.optim.Optimizer:
    optim_type = args.optimizer
    if optim_type == 'adam':
        return torch.optim.Adam(params = model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    elif optim_type == 'sgd':
        return torch.optim.SGD(params = model.parameters(), lr = args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif optim_type == 'adamw' : 
        return torch.optim.AdamW(params = model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    else:
        raise Exception()