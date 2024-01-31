#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

import os

import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from Dataset.SequenceDatasets import dataset
from Dataset.sequence_aug import *

root_path = r'D:/P_new/PU_dataset'
signal_size = 1024

WC = ["N15_M07_F10", "N15_M01_F10", "N15_M07_F04"]
RDBdata = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23','KB24','KB27', 'KI14','KI17','KI21']
label3 = [i for i in range(len(RDBdata))]

def get_files(root, N, input_kind='fft'):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for i in range(len(N)):
        state = WC[N[i]]  # WC[0] can be changed to different working states
        print("the PU data is {}".format(RDBdata))
        for k in tqdm(range(len(RDBdata))):
            for w3 in range(1):
                name3 = state + "_" + RDBdata[k] + "_" + str(w3 + 1)
                path3 = os.path.join('/tmp', root, RDBdata[k], name3 + ".mat")
                data3, lab3 = data_load(path3, name=name3, label=label3[k], input_kind=input_kind)
                data += data3
                lab += lab3
        #
        # for j in tqdm(range(len(ORdata))):
        #     for w2 in range(20):
        #         name2 = state + "_" + ORdata[j] + "_" + str(w2 + 1)
        #         path2 = os.path.join(root, ORdata[j], name2 + ".mat")
        #         data2, lab2 = data_load(path2, name=name2, label=label[1], input_kind=input_kind)
        #         data += data2
        #         lab += lab2
        #
        # for j in tqdm(range(len(IRdata))):
        #     for w2 in range(20):
        #         name3 = state + "_" + IRdata[j] + "_" + str(w2 + 1)
        #         path3 = os.path.join(root, IRdata[j], name3 + ".mat")
        #         data3, lab3 = data_load(path3, name=name3, label=label[2], input_kind=input_kind)
        #         data += data3
        #         lab += lab3

    return [data, lab]


def data_load(filename, name, label, input_kind='fft'):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  # Take out the data

    fl = fl.reshape(-1, )

    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = fl[start:end]
        if input_kind == 'fft':
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1, 1)
        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


class PU(object):
    inputchannel = 1
    num_classes = len(RDBdata)

    def __init__(self, TL_list, data_path, TL_Task, seed_run=None, TL_kind=None,
                 data_name='PU', norm_kind="mean-std", input_kind='fft', **kwargs):
        self.TL_kind = TL_kind
        self.TL_list = TL_list
        self.data_name = data_name
        self.data_path = data_path
        self.source_N = [int(x) for x in str(TL_Task[0])]
        self.target_N = [int(x) for x in str(TL_Task[1])]
        self.norm_kind = norm_kind
        self.input_kind = input_kind
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.norm_kind),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),

        }

    def data_generator(self):
        list_data = get_files(self.data_path, self.source_N, self.input_kind)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        source_data = dataset(list_data=data_pd, transform=self.data_transforms['train'])

        list_data = get_files(self.data_path, self.target_N, self.input_kind)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        target_data = dataset(list_data=data_pd, transform=self.data_transforms['train'])
        return source_data, target_data


if __name__ == '__main__':
    PU_Data = PU(data_path=root_path, transfer_task=[0, 1])
    source_data, target_data = PU_Data.data_generator()