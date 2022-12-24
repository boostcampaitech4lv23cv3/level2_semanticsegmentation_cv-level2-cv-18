import os
import torch
import argparse

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
    file_name = args.save_name.format(model=model.__class__.__name__)

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
    os.makedirs(saved_dir, exist_ok=True)
    if not os.path.isdir(saved_dir):     
        os.makedirs(saved_dir, exist_ok=True)
    output_path = os.path.join(saved_dir, file_name)

    # save
    torch.save(save_obj, output_path)
    print(f"The model has been saved at {output_path}")


def load_model(args:argparse.Namespace):
    '''
    [args]
        args는 load_path과 load_name을 반드시 가져야 합니다.
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
    load_dir = args.load_path
    file_name = args.load_name


    load_path = os.path.join(load_dir, file_name)

    # save
    load_obj = torch.load(load_path)
    print(f"The date has been loaded from {load_path}")
    
    return load_obj