import math
import wandb
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from munch import Munch
import torch

def get_sweep_config():
    sweep_config = dict(
        name = 'bayes-test',
        method = 'grid',
        metric = dict(
            name = 'val/mIoU',
            goal = 'maximize'
        ))
    # 이 항목에서 추가하실 것들을 넣고 수정하시면 됩니다!
    sweep_config['parameters'] = dict(
        # optimizer=dict(values=['Adam','sgd']),
        lr=dict(values=[0.0001]),
        epoch=dict(values=[20]),
        batch_size=dict(values=[8]),
        )
   
    return sweep_config


def sweep_init(args):
    wandb.init(
        name='name',
        group='sweep_name',
        reinit=True,
    )
    
def run_sweep():
    wandb.init(config=sweep_config, 
               entity="light-observer",
               project='sweep_test')
    w_config = wandb.config

def wandb_init(args):
    wandb.init(
        project='sweep_test',
        entity="light-observer",
        name='test',
        reinit=True,
        config=args.__dict__,
    )

def get_sweep_id(sweep_config):
    sweep_id = wandb.sweep(
        sweep=sweep_config,
    )
    return sweep_id

def init_wandb(args:Namespace):
    if args.wandb_enable :
        wandb.init(
            entity="light-observer",
            project="Trash Segmentation",
            name=args.model if args.wandb_name == None else args.wandb_name,
            config=args.__dict__ # track hyperparameters and run metadata
        )

def concat_config(args, config):
    config = Munch(config)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['wandb_enable'] = args.wandb_enable
    config['wandb_name'] = args.wandb_name
    
    config['notion_post_enable'] = args.notion_post_enable
    config['notion_post_name'] = args.notion_post_name
    config['seed'] = args.seed
    config['config_path'] = args.config_path
    config['config_file'] = args.config_file
    config['data_path'] = args.data_path
    config['save_path'] = args.save_path

    
    config['model'] = args.model
    config['batch_size'] = args.batch_size
    config['val_every'] = args.val_every
    config['epoch'] = args.epoch
    config['lr'] = args.lr
    config['weight_decay'] = args.weight_decay
    config['scheduler_step'] = args.scheduler_step


    config['scheduler_gamma'] = args.scheduler_gamma
    config['valid_img'] = args.valid_img
    config['sweep'] = args.sweep
    
    return config