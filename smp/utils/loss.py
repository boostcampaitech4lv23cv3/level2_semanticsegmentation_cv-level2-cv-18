from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
from typing import Optional

def get_loss(args:Namespace) -> nn.Module:
    loss = nn.CrossEntropyLoss()
    if args.loss == 'focal':
        if args.fl_alpha == 'basic':
            f_alpha = torch.tensor([1.44,44.33,11.04,139.99,111.34,130.32,34.69,66.27,8.56,2162.59,187.92]).to(args.device)
        elif args.fl_alpha == 'max':
            f_alpha = torch.tensor([0.0007, 0.0205, 0.0051, 0.0647, 0.0515, 0.0603, 0.016 , 0.0306, 0.004 , 1.0, 0.0869]).to(args.device)
        elif args.fl_alpha == 'avg':
            f_alpha = torch.tensor([0.0054, 0.1682, 0.0419, 0.5313, 0.4225, 0.4946, 0.1317, 0.2515, 0.0325, 8.2072, 0.7132]).to(args.device)
        elif args.fl_alpha == 'miou':
            # f_alpha = torch.tensor([0.1, 0.6, 0.3, 0.6, 0.5, 0.5, 0.6, 0.3, 0.2, 0.4, 0.5]).to(args.device)
            f_alpha = torch.tensor([0.2, 1.2, 0.6, 1.2, 1.2, 1.0, 1.2, 0.6, 0.4, 2.0, 1.4]).to(args.device)
        else:
            f_alpha = float(args.fl_alpha)
        loss = FocalLoss(f_alpha, gamma = 2.0, reduction = 'mean', ls=args.ls)
    elif args.loss == 'dice':
        loss = smp.losses.DiceLoss(mode='multiclass')
    elif args.loss == 'jaccard':
        loss = smp.losses.JaccardLoss(mode='multiclass')
    elif args.loss == 'cross_entropy':
        loss = loss
    else:
        print(args.loss, ' is not supported option.')
    print(' * loss : ', loss.__class__.__name__)
    return loss


def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    ignore_index=255,
) -> torch.Tensor:
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
    
    return ret


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(input:torch.Tensor, 
                target:torch.Tensor, 
                alpha, gamma:float, 
                reduction:str, eps:float, 
                ignore_index:int,
                ls: float = 0.1
                ):
    """
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0) # B
    
    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]
    
    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)       
        

    # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps
    
    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)
    
    # label smoothing
    target_one_hot = ((1.0 - ls) * target_one_hot) + ls / target_one_hot.shape[1]
    m = torch.min(target_one_hot)
    m = torch.max(target_one_hot)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)
    
    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)
    
    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma:float = 2.0, reduction:str = 'mean', eps:float = 1e-8, ignore_index:int=30, ls:float=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index
        self.ls = ls

    def forward(self, input, target):
        return focal_loss(
            input=input, target=target, 
            alpha=self.alpha, 
            gamma=self.gamma, 
            reduction=self.reduction, 
            eps=self.eps, ignore_index=self.ignore_index,
            ls=self.ls
            )