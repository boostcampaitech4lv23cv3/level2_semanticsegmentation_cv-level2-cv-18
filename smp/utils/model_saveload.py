import os
import torch
import argparse
import json
from models import *
from torch.optim.optimizer import Optimizer

def make_directory(path:str):
    fd = os.path.dirname(path)
    if(fd == ''):
        return
    else:
        os.makedirs(fd, exist_ok=True)
        if not os.path.isdir(fd):     
            os.makedirs(fd, exist_ok=True)

def save_args(args:argparse.Namespace, path:str):
    make_directory(path=path)
    with open(path, 'w') as outfile:
        json.dump(args.__dict__, outfile, indent = 4, sort_keys = True)

def save_model(best_loss:float, best_epoch:int, args:argparse.Namespace, model, optimizer, criterion, time, include_optimizer_state:bool = False):
    '''
    [object structure]
        'start_time' : time,
        'best_loss' : best_loss,
        'best_epoch' : best_epoch,
        'model_name' : model.__class__.__name__,
        'model_state' : model.state_dict(),
        'optimizer_name' : optimizer.__class__.__name__,
        'criterion_name' : criterion.__class__.__name__,
        'include_optimizer_state' : include_optimizer_state,
        'args' : args
        ('optimizer_state' : optimizer.state_dict()) # included when include_optimizer_state = True
    '''

    # get params from args
    saved_dir = args.save_path
    file_name = args.save_name.format(model=model.__class__.__name__, time=time)

    # make dict
    save_obj = {
        'start_time' : time,
        'best_loss' : best_loss,
        'best_epoch' : best_epoch,
        'model_name' : model.__class__.__name__,
        'model_state' : model.state_dict(),
        'optimizer_name' : optimizer.__class__.__name__,
        'criterion_name' : criterion.__class__.__name__,
        'include_optimizer_state' : include_optimizer_state,
        'args' : args
    }
    if include_optimizer_state :
        save_obj['optimizer_state'] = optimizer.state_dict()

    # set directory
    output_path = os.path.join(saved_dir, file_name)
    make_directory(path = output_path)
    save_args(args=args, path=output_path+'.json')
    
    # save
    torch.save(save_obj, output_path)
    
    print(f"The data has been saved at {output_path}")


def load_model(args:argparse.Namespace):
    '''
    [args]
        args는 load_path과 load_file을 반드시 가져야 합니다.
    '''

    # get params from args
    load_dir = args.load_path
    file_name = args.load_file


    load_path = os.path.join(load_dir, file_name)

    # save
    load_obj = torch.load(load_path)
    print(f"The data has been loaded from {load_path}")

    return CheckpointData(load_obj)

class CheckpointData:
    def __init__(self, loaded_data:dict) -> None:
        self.raw_data = loaded_data
        self.start_time = loaded_data['start_time']
        self.best_loss = loaded_data['best_loss']
        self.best_epoch = loaded_data['best_epoch']
        self.model_name = loaded_data['model_name']
        self.model_state = loaded_data['model_state']
        self.optimizer_name = loaded_data['optimizer_name']
        self.criterion_name = loaded_data['criterion_name']
        self.include_optimizer_state = loaded_data['include_optimizer_state']
        self.args = loaded_data['args']
        self.optimizer_state = None
        if self.include_optimizer_state:
            self.optimizer_state = loaded_data['optimizer_state']

    def create_model_from_data(self) -> torch.nn.Module:
        model = eval('{}()'.format(self.model_name))
        model.load_state_dict(self.model_state)
        return model

    def create_optimizer_from_data(self) -> Optimizer:
        optimizer = eval('{}()'.format(self.optimizer_name))
        if self.include_optimizer_state:
            optimizer.load_state_dict(self.optimizer_state)
        return optimizer

    def describe_data(self):
        print(' - CheckpointData')
        print('   > start_time : ', self.start_time)
        print('   > best_loss : ', self.best_loss)
        print('   > best_epoch : ', self.best_epoch)
        print('   > model_name : ', self.model_name)
        print('   > optimizer_name : ', self.optimizer_name)
        print('   > criterion_name : ', self.criterion_name)