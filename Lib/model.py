#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

import torch.nn as nn
import numpy as np
import torch.optim as optim
from Model_Zoos.Classifier import Bottleneck, Classifier
import Model_Zoos


def get_model(
        model_name,
        num_classes,
        bottleneck=True,
        bottleneck_num=256,
        model_type='linear',
        temp=1,
        Dropout=False, cfg=None, net_F=None, **kwargs):
    if net_F is None:
        assert model_name in ['ResNet18']
        model_ = getattr(Model_Zoos, model_name)
        netF = model_(cfg=cfg)

    if bottleneck:
        netB = Bottleneck(in_num=netF.output_num,
                          bottleneck_num=bottleneck_num, Dropout=Dropout)
        netC = Classifier(
            class_num=num_classes,
            bottleneck_dim=bottleneck_num,
            type=model_type,
            temp=temp)
        model = nn.Sequential(netF, netB, netC)
    else:
        netC = Classifier(
            class_num=num_classes,
            bottleneck_dim=bottleneck_num,
            type=model_type,
            temp=temp)
        model = nn.Sequential(netF, netC)

    return model




def get_opt_parameters(parameters, args, kind='src'):
    # args is Cfg.Opt
    if kind == 'src':
        if args.name == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=args.lr_src,
                momentum=args.momentum,
                weight_decay=args.weight_decay_src,
                nesterov=args.nesterov)
        elif args.name == 'adamw':
            optimizer = optim.AdamW(
                parameters,
                lr=args.lr_src,
                weight_decay=args.weight_decay_src)
        elif args.name == 'adam':
            optimizer = optim.Adam(
                parameters,
                lr=args.lr_src,
            )
        else:
            raise Exception("optimizer not implement")

    elif kind == 'tar':
        if args.name == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=args.lr_tar,
                momentum=args.momentum,
                weight_decay=args.weight_decay_tar,
                nesterov=args.nesterov)
        elif args.name == 'adamw':
            optimizer = optim.AdamW(
                parameters,
                lr=args.lr_tar,
                weight_decay=args.weight_decay_tar)
        elif args.name == 'adam':
            optimizer = optim.Adam(
                parameters,
                lr=args.lr_tar,
            )
        else:
            raise Exception("optimizer not implement")

    return optimizer



def get_opt(model, args, kind='src'):
    # args is Cfg.Opt
    if kind == 'src':
        if args.opt == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr_src,
                momentum=args.momentum,
                weight_decay=args.weight_decay_src,
                nesterov=args.nesterov)
        elif args.opt == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr_src,
                weight_decay=args.weight_decay_src)
        elif args.opt == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr_src,
            )
        else:
            raise Exception("optimizer not implement")

    elif kind == 'tar':
        if args.opt == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr_tar,
                momentum=args.momentum,
                weight_decay=args.weight_decay_tar,
                nesterov=args.nesterov)
        elif args.opt == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr_tar,
                weight_decay=args.weight_decay_tar)
        elif args.opt == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr_tar,
            )
        else:
            raise Exception("optimizer not implement")

    return optimizer


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


if __name__ == '__main__':
    print()