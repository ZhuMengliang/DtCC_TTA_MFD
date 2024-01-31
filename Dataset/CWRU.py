#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import os

import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from Dataset.SequenceDatasets import dataset
from Dataset.sequence_aug import *

root_path = r"D:/P_new/CWRU_dataset"
dataname = {0: ["97.mat", "105.mat", "118.mat", "130.mat", "169.mat",
                "185.mat", "197.mat", "209.mat", "222.mat", "234.mat"],  # 1797rpm
            1: ["98.mat", "106.mat", "119.mat", "131.mat", "170.mat",
                "186.mat", "198.mat", "210.mat", "223.mat", "235.mat"],  # 1772rpm
            2: ["99.mat", "107.mat", "120.mat", "132.mat", "171.mat",
                "187.mat", "199.mat", "211.mat", "224.mat", "236.mat"],  # 1750rpm
            3: ["100.mat", "108.mat", "121.mat", "133.mat", "172.mat",
                "188.mat", "200.mat", "212.mat", "225.mat", "237.mat"]}  # 1730rpm

datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data",
               "48k Drive End Bearing Fault Data", "Normal Baseline Data"]

label = [i for i in range(0, 10)]
signal_size = 1024


def get_files(root, N, input_kind='fft'):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):  # N:the list of task indexes
        for n in tqdm(range(len(dataname[N[k]]))):  # for each class
            path1 = os.path.join(root, dataname[N[k]][n])
            data1, lab1 = data_load(path1, dataname[N[k]][n], label=label[n], input_kind=input_kind)
            data += data1
            lab += lab1
    return [data, lab]


def data_load(filename, axisname, label, input_kind='fft'):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    input_kind='fft','time'
    '''
    axis = ["_DE_time", "_FE_time", "_BA_time"]
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
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
        elif input_kind == 'phase':
            x = np.fft.fft(x)
            x = np.angle(x)

        x = x.reshape(-1, 1)
        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size
    return data, lab


# --------------------------------------------------------------------------------------------------------------------
class CWRU(object):
    num_classes = 10
    inputchannel = 1

    def __init__(self, TL_list, data_path, TL_Task, TL_kind=None, seed_run=None, data_name='CWRU',
                 norm_kind="mean-std", input_kind='fft', **kwargs):
        self.TL_kind = TL_kind
        self.TL_list = TL_list
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
    CWRU_Data = CWRU(data_path=root_path, TL_Task=[0, 2])
    source_data, target_data = CWRU_Data.data_generator()
    # source_data.seq_data:list

