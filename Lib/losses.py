#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

import torch
import torch.nn as nn
import torch.nn.functional as F


def IM_loss(outputs):
    # Mutual Information Maximizing: H(Y|X)-H(Y)
    im_loss = Entropy_loss(outputs) - Div_loss(outputs)
    return im_loss


def Div_loss_(msoftmax):
    div_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return div_loss

def Div_loss(outputs):
    softmax_out = nn.Softmax(dim=1)(outputs)
    msoftmax = softmax_out.mean(dim=0)
    div_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return div_loss


def Entropy_loss(outputs):
    
    softmax_out = nn.Softmax(dim=1)(outputs)
    entropy_loss = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out + 1e-5), dim=1))
    return entropy_loss


def BNM_loss(outputs, kind='fast'):
    if kind == 'fast':
        out = nn.Softmax(dim=1)(outputs)
        list_svd, _ = torch.sort(torch.sqrt(torch.sum(torch.pow(outputs, 2), dim=0)), descending=True)
        bnm_loss = - torch.mean(list_svd[:min(outputs.shape[0], outputs.shape[1])])
    else:
        softmax_out = nn.Softmax(dim=1)(outputs)
        _, s_tgt, _ = torch.svd(softmax_out)
        bnm_loss = -torch.mean(s_tgt)
    
    return bnm_loss


class CrossEntropyLabelSmooth(nn.Module):
    
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True, temperature=1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.temperature = temperature
    
    def forward(self, inputs, targets):
        inputs = inputs * self.temperature
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


class SymmetricCrossEntropyLoss(nn.Module):
    """The Symmetric Cross Entropy loss.
    It has been proposed in `Symmetric Cross Entropy for Robust Learning
    with Noisy Labels`_.
    .. _Symmetric Cross Entropy for Robust Learning with Noisy Labels:
        https://arxiv.org/abs/1908.06112
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, device=torch.device("cuda")):
        """
        Args:
            alpha(float):
                corresponds to overfitting issue of CE
            beta(float):
                corresponds to flexible exploration on the robustness of RCE
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
    
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``input_`` and ``target`` tensors.
        Args:
            input_: input tensor of size: logits
                (batch_size, num_classes)
            target: target tensor of size (batch_size), where
                values of a vector correspond to class index
        Returns:
            torch.Tensor: computed loss
        """
        num_classes = input_.shape[1]
        target_one_hot = F.one_hot(target, num_classes).float()
        assert target_one_hot.shape == input_.shape
        input_ = F.softmax(input_,dim=1)
        input_ = torch.clamp(input_, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0).to(self.device)
        
        cross_entropy = (-torch.sum(target_one_hot * torch.log(input_), dim=1)).mean()
        reverse_cross_entropy = (
            -torch.sum(input_ * torch.log(target_one_hot), dim=1)
        ).mean()
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10, epsilon=6):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=10 ** (-self.epsilon), max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=10 ** (-self.epsilon), max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
        
        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss


def nl_criterion(output, y):
    output = torch.log(torch.clamp(1. - F.softmax(output, dim=1), min=1e-5, max=1.))
    l = F.nll_loss(output, y, reduction='mean')
    return l




if __name__ == '__main__':
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    nl_criterion(input,target,num_class=3)