#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch
import numpy as np
import torch.nn as nn
from Lib.train_utils import cal_acc

def Model_Prune(model_src, cfg, self):
    total = 0
    for m in model_src[0].modules():
        if isinstance(m, nn.BatchNorm1d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model_src[0].modules():
        if isinstance(m, nn.BatchNorm1d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)  # 从小到大排序
    A = 1 - cfg.PR/2
    B = cfg.PR/2

    high = torch.quantile(y, A)
    low = torch.quantile(y, B)
    # 取low~high这一范围内的值

    thre_index = int(total * cfg.PR)
    thre = y[thre_index]

    pruned = 0
    cfg_model = []
    cfg_mask = []

    for k, m in enumerate(model_src[0].modules()):
        if isinstance(m, nn.BatchNorm1d) and k not in [27, 43, 59]:
            # ResNetv1 的downsample 那里有BN层，该如何处理？手动记下来downsample中的BN所在的索引k值
            weight_copy = m.weight.data.abs().clone()
            mask_low = weight_copy.gt(low).float().cuda()  # 保留下来的索引mask
            mask_high = weight_copy.lt(high).float().cuda()
            mask = mask_low*mask_high
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)  # 保留下来的参数
            m.bias.data.mul_(mask)  # 保留下来的参数
            cfg_model.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))

    pruned_ratio = pruned / total * 100
    acc_t = cal_acc(self.dataloaders["target_data"], model_src, self.device)[0]
    print(f"the prune ratio is {pruned_ratio:.2f} and the acc_t is {acc_t:.2f}")
    return cfg_model, cfg_mask






