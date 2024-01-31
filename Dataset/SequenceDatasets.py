#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
from torch.utils.data import Dataset

from Dataset.sequence_aug import *


class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform

    def __len__(self):  # 返回长度
        return len(self.seq_data)

    def __getitem__(self, item):  # 返回索引
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            # 这里是对每个sliced的1-D样本做归一化
            return seq, label, item
