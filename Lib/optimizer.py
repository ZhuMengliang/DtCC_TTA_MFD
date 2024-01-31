#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch
import torch.optim as optim
import numpy as np


# ......这里是否可以直接写入配置文件
def get_optimizer(model, args, kind='src'):
    # args is Cfg.Opt
    if kind == 'src':
        if args.name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr_src,
                momentum=args.momentum,
                weight_decay=args.weight_decay_src,
                nesterov=args.nesterov)
        elif args.name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr_src,
                weight_decay=args.weight_decay_src)
        elif args.name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr_src,
            )
        else:
            raise Exception("optimizer not implement")
    
    elif kind == 'tar':
        if args.name == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr_tar,
                momentum=args.momentum,
                weight_decay=args.weight_decay_tar,
                nesterov=args.nesterov)
        elif args.name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr_tar,
                weight_decay=args.weight_decay_tar)
        elif args.name == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr_tar,
            )
        else:
            raise Exception("optimizer not implement")
    
    return optimizer


# # 这里的**kwargs是精髓，传入的是形参字典
# def get_optimizer(name, params, **kwargs):
#     name = name.lower()
#     optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
#     optim_cls = optimizers[name]
#
#     return optim_cls(params, **kwargs)


def get_lr_scheduler(optimizer, args, epoch_train, kind):
    if kind == 'src':
        lr = args.lr_src
    elif kind == 'tar':
        lr = args.lr_tar
    else:
        raise NotImplementedError
    
    def lambda1(epoch):
        return (1 + epoch / epoch_train) ** (-1 / np.log10(2))
    
    def lambda2(epoch):
        return (1 + 10 * epoch / epoch_train) ** (-0.75)
        # 这个是TL论文里面常用的学习率衰减策略，
        # 通常配合SGD，lr=1e-2，weight decay=1e-3使用
    
    step_list = list(np.arange(epoch_train // 3, epoch_train + 1, epoch_train // 3))
    scheduler_dict = {
        'cosine': optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epoch_train,
            eta_min=lr / 10,
        ),
        'step': optim.lr_scheduler.StepLR(
            optimizer,
            step_size=epoch_train // 2,
            gamma=0.1),
        'designed': optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda1),
        
        'paper': optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda2),
        
        'mult_step': optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=step_list,
            gamma=0.1),
        'fix': None,
        'exp': optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.99)}
    scheduler = scheduler_dict.get(args.lr_scheduler, None)
    return scheduler