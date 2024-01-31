#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

def Tsallis_entropy(predictions: torch.Tensor, alpha=2.0, reduction='mean'):
    if alpha == 1.0:
        epsilon = 1e-8
        H = -predictions * torch.log(predictions + epsilon)
        H = H.sum(dim=1)
    else:
        H = 1 / (alpha - 1) * (1 - torch.sum(predictions ** alpha, dim=-1))
    if reduction == 'mean':
        return H.mean()
    elif reduction == 'sum':
        return H.sum()
    else:
        return H



def PCL(features_certain, features_uncertain, prototypes, labels, temp=1):
    features_certain = F.normalize(features_certain, dim=1)
    if features_uncertain is not None:
        features_uncertain = F.normalize(features_uncertain, dim=1)
        con_features = torch.cat([prototypes, features_uncertain.T], dim=1)
    else:
        con_features = prototypes
    logits = torch.mm(features_certain, con_features) / temp
    # for numeric stability
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    A = torch.exp(logits)
    B = A.sum(dim=1, keepdim=True) + 1e-10
    loss2 = F.nll_loss(torch.log(A / B), labels)
    return loss2



def NCL(features_uncertain, supports, scores, neighbor_K, probs_uncertain, temp=1):
    distance = F.normalize(features_uncertain.detach().clone(), dim=1) \
               @ F.normalize(supports, dim=1).t().detach().clone()
    _, index_near = torch.topk(distance, dim=-1, largest=True, k=neighbor_K)
    P_near = scores[index_near]  # M*NN*K

    n_uncertain = features_uncertain.size(0)

    index_far = torch.arange(n_uncertain).expand((n_uncertain, n_uncertain))
    index_ = (index_far - torch.diag_embed(index_far.diag() + 1) != -1)
    index_far = index_far[index_].reshape(n_uncertain, -1)

    P_far = probs_uncertain[index_far]  # B*(B-1)*C#
    P_uncertain = probs_uncertain.unsqueeze(1)  # B*1*C
    P_NF = torch.cat([P_near, P_far], dim=1).permute(0, 2, 1)
    P = torch.bmm(P_uncertain, P_NF) / temp
    P = P - torch.max(P, dim=-1, keepdim=True)[0]
    A = torch.exp(P)
    B = A.sum(dim=2, keepdim=True) + 1e-10

    # Inspired by supervised contrastive learning
    C = -torch.log((A / B)[:, :, 0:neighbor_K].mean(dim=2))
    loss = C.mean()
    
    return loss



def Div_loss(probs):
    msoftmax = probs.mean(dim=0)
    div_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return div_loss


def Balance_P(prob_predictions, index_certain, B_kind=1):

    n_c = prob_predictions.size(1)
    frequency = torch.tensor(
        [(prob_predictions[index_certain, :].argmax(dim=1) == i).float().sum() for i in range(n_c)])\
        .reshape(1, -1).to(prob_predictions.device)
    prob_predictions_b = prob_predictions / (frequency + 1)
    return prob_predictions_b


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('Model Size：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)