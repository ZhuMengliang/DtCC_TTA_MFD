#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F
from Model_Zoos.model_utils import GradientReverseLayer

grl = GradientReverseLayer()

class Bottleneck(nn.Module):
    def __init__(self, in_num=512, bottleneck_num=256,Dropout=False):
        super(Bottleneck, self).__init__()
        if Dropout:
            self.fc = nn.Sequential(nn.Linear(in_num, bottleneck_num), nn.BatchNorm1d(bottleneck_num),
                                nn.ReLU(inplace=True), nn.Dropout())
        else:
            self.fc = nn.Sequential(nn.Linear(in_num, bottleneck_num), nn.BatchNorm1d(bottleneck_num),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.fc(x)
        return x
class Classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear", temp=2):
        super(Classifier, self).__init__()
        self.temp = temp
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            # cosine similarity-based classifier architecture
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)
    
    def forward(self, x, GRL=False, coeff=1):
        if GRL:
            x=grl(x,coeff)
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = F.normalize(w, dim=1, p=2)
            x = F.normalize(x, dim=1, p=2)
            x = F.linear(x, w)*self.temp
        else:
            x = self.fc(x)
        return x
    
class Dist_Classifier(nn.Module):
    # A CLOSER LOOK AT FEW-SHOT CLASSIFICATION
    # https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py
    
    def __init__(self, class_num, bottleneck_dim=256, class_wise_learnable_norm=True, temp=2):
        super(Dist_Classifier, self).__init__()
        self.L = nn.Linear(bottleneck_dim, class_num, bias=False)
        self.class_wise_learnable_norm = class_wise_learnable_norm  #See the issue#4&8 in the github
        
        if self.class_wise_learnable_norm:
            self.L = weightNorm(self.L, name="weight")
        self.temp=temp
            
    def forward(self, x):
        x = F.normalize(x, dim=1, p=2)
        if not self.class_wise_learnable_norm:
            self.L.weight.data = F.normalize(self.L.weight.data) # 源代码有问题的
        cos_dist = self.L(x) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.temp * cos_dist
        return scores
    
class Predictor(nn.Module):
    # 这是SS-DA里面常用到的结果，虽然实际上是有点问题的，因为最终计算出来的并不是余弦值。
    def __init__(self, class_num, bottleneck_dim=256, temp=2):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
        self.num_class = class_num
        self.temp = temp
    def forward(self, x, GRL=False, coeff=1):
        if GRL:
            x = grl(x, coeff)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out
    
if __name__ == '__main__':
    Dist_C = Dist_Classifier(class_num=10, bottleneck_dim=50, class_wise_learnable_norm=False, temp=1)
    input = torch.randn(100, 50)
    output = Dist_C(input)