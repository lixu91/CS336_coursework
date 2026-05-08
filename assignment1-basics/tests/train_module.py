import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np

def crs_enty_ls(pre_logits:torch.Tensor, targets:torch.Tensor ):


    max_logits = pre_logits.max(dim=-1,keepdim= True).values
    shifted_logit = pre_logits - max_logits
    minus_term = - shifted_logit.gather(dim =-1 ,index = targets.unsqueeze(-1)).squeeze(-1)
    plus_term = torch.log(torch.exp(shifted_logit).sum(dim = -1))
    loss = (minus_term + plus_term).sum() / targets.numel()
    return loss


"""
自定义优化器需要继承 Optimizer父类, super().__init__(params, default), params为需要调节的model参数,default为字典类型, 记录lr moment等超参数， 
父类会初始化self.param_groups and self.states。
self.param_groups: type list， list中每个元素为一个字典，字典中包括了model参数，超参数等
self.state 给每个参数保存历史信息的地方 也为字典类型， key为参数，value为包含超参数的字典

"""

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr:float = 1e-3, betas:tuple =(0.9,0.999), eps = 1e-8, weight_decay=0.01):
        default = {'lr' : lr, 'betas' : betas, 'eps' : eps, 'weight_decay' : weight_decay}
        super().__init__(params, default)
    
    def step(self,closure: Optional[Callable] = None ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['betas'][0]
            beta2 = group['betas'][1]
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get('t',1)
                lr_v = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                p.data = p.data - lr*weight_decay*p.data
                m = state.get('m',0)
                v = state.get('v',0)
                m = beta1 * m + (1-beta1)* p.grad.data
                v = beta2 * v + (1- beta2)* p.grad.data ** 2
                p.data = p.data - lr_v * m / (torch.sqrt(v) + eps)
                state['m'] = m
                state['v'] = v
                state['t'] = t+1
        return loss


def cos_annealing(t, lr_max, lr_min, T_w, T_c):
    if t< T_w:
        return t* lr_max / T_w
    elif t<= T_c:
        return  lr_min + 0.5*(1 + math.cos((t-T_w)* math.pi / ( T_c - T_w)))*(lr_max - lr_min)
    else: return lr_min

def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float)->None :
    eps = 1e-6
    l2 = 0
    for p in parameters:
        if p.grad is None:
            continue
        l2 += torch.sum(p.grad**2)
    l2 = torch.sqrt(l2)
    if l2 > max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad = p.grad* max_l2_norm / (l2 + eps)


def data_loader(dataset, batch_size: int, context_length: int, device: str):
    index_list = np.random.randint(0, len(dataset) - context_length, size = batch_size)
    sampled_dataset = np.empty((batch_size, context_length))
    target = np.empty((batch_size, context_length))
    for batch_idx, contex_idx in enumerate(index_list):
        sampled_dataset[batch_idx] = dataset[contex_idx :contex_idx + context_length] 
        target[batch_idx] = dataset[contex_idx+1 :contex_idx+1 + context_length]
    s = torch.tensor(sampled_dataset,device =device,dtype = torch.long)
    t = torch.tensor(target,device =device, dtype = torch.long)
    return (s,t)

def save_checkpoint(model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    overall_state = {'model' : model_state, 'optimizer' : optimizer_state, "iter":iteration}
    torch.save(overall_state, out)

def load_checkpoint(src, model,optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['iter']