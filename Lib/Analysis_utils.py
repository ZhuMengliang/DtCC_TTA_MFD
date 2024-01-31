#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from typing import Optional


def get_features(model, dataloaders, device=torch.device('cuda')):
    model.eval()
    target_dataloader = dataloaders['target_data']
    source_dataloader = dataloaders['source_data']
    with torch.no_grad():
        start_test = True
        iter_src = iter(source_dataloader)
        iter_tar = iter(target_dataloader)
        for _ in range(len(iter_src)):
            data = iter_src.next()
            inputs = data[0]
            inputs = inputs.to(device)
            fea_src = model[1](model[0](inputs))
            if start_test:
                all_fea_src = fea_src.float().cpu()
                start_test = False
            else:
                all_fea_src = torch.cat((all_fea_src, fea_src.float().cpu()), 0)
        start_test = True
        iter_tar = iter(target_dataloader)
        for _ in range(len(iter_tar)):
            data = iter_tar.next()
            inputs = data[0]
            inputs = inputs.to(device)
            fea_tar = model[1](model[0](inputs))
            if start_test:
                all_fea_tar = fea_tar.float().cpu()
                start_test = False
            else:
                all_fea_tar = torch.cat((all_fea_tar, fea_tar.float().cpu()), 0)
    
    all_label_tar = torch.Tensor(dataloaders["target_data"].dataset.labels)
    all_label_src = torch.Tensor(dataloaders["source_data"].dataset.labels)
    
    
    return all_fea_tar, all_fea_src, all_label_tar, all_label_src



def tsne_plot_features(features,labels, filename='tsne.jpg', norm=True, scale=None):
    features = features.numpy()
    labels = labels
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import seaborn as sns
    import numpy as np
    from sklearn.manifold import TSNE

    if norm:
        min_value, max_value = np.min(features, 0), np.max(features, 0)
        features = (features - min_value) / ((max_value - min_value) + 1e-8)
        if scale:
            features = scale * features
    plt.rcParams['savefig.dpi'] = 480  # 图片像素
    plt.rcParams['figure.dpi'] = 480  # 分辨率
    plt.rcParams['figure.figsize'] = (8.0, 4.0)
    

    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        random_state=42)
    features_tsne = tsne.fit_transform(features)
    import pandas as pd
    df = pd.DataFrame(features_tsne[:, 0], columns=['dim0'])
    df['dim1'] = features_tsne[:, 1]
    df['y'] = labels
    df['size'] = 5 * np.ones(features_tsne.shape[0])
    plt.figure(figsize=(8, 6))
    axe = sns.scatterplot(
        x="dim0", y="dim1",
        hue="y",
        palette=sns.color_palette("hls", torch.unique(labels).size()[0]),
        data=df,
        alpha=0.8,
        size='size'
    )
    axe.legend_.remove()  # 去掉图例
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    axe.spines['bottom'].set_visible(False)
    axe.spines['left'].set_visible(False)
    plt.xticks([])  # 去掉刻度
    plt.yticks([])
    axe.set(xticklabels=[])
    axe.set(xlabel=None)
    axe.set(yticklabels=[])
    axe.set(ylabel=None)
    plt.savefig(filename, format='png', dpi=480)
    # plt.show()
    plt.close()
    plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
    plt.clf()  # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
    plt.close()  # 关闭 window，如果没有指定，则指当前 window。
    del plt  # 这个地方要把plt这个变量给删掉，不然下面主函数调用的时候，一旦使用了plt.show()，也会把这里的图给显示出来。


def tsne_plot_domains(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b', norm=True, scale=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    import numpy as np
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    if norm:
        min_value, max_value = np.min(source_feature, 0), np.max(source_feature, 0)
        source_feature = (source_feature - min_value) / (max_value - min_value)
        min_value, max_value = np.min(target_feature, 0), np.max(target_feature, 0)
        target_feature = (target_feature - min_value) / (max_value - min_value)
        if scale:
            source_feature = scale * source_feature
            target_feature = scale * target_feature
    
    features = np.concatenate([source_feature, target_feature], axis=0)
    # map features to 2-d using TSNE
    from sklearn.manifold  import TSNE
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(features)
    
    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
    
    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, format='jpg', dpi=480, bbox_inches='tight', pad_inches=0.1)


def A_distance_svm(source_feature: torch.Tensor, target_feature: torch.Tensor, progress=True,):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.
    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier
    Returns:
        :math:`\mathcal{A}`-distance
    """
    import numpy as np
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    ns = source_feature.size()[0]
    nt = target_feature.size()[0]
    index = np.arange(ns)
    # np.random.shuffle(index)
    # index_train = index[:int(0.8*ns)]
    # index_test = index[int(0.8*ns):]
    index_train = index[:int(0.5*ns)]
    index_test = index[int(0.5 * ns):]

    trainX = torch.cat([source_feature[:ns//2], target_feature[:ns//2]], dim=0)
    trainY = torch.cat([source_label[:ns//2], target_label[:ns//2]], dim=0).squeeze()
    testX = torch.cat([source_feature[ns//2:], target_feature[ns//2:]], dim=0)
    testY = torch.cat([source_label[ns//2:], target_label[ns//2:]], dim=0).squeeze()
    C_list = [0.01,0.1,1,10]

    best_risk = 1.0
    from sklearn import svm
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(trainX, trainY)
        train_risk = np.mean(clf.predict(trainX) != trainY.numpy())
        test_risk = np.mean(clf.predict(testX) != testY.numpy())
        
        if progress:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))
        
        if test_risk > .5:
            test_risk = 1. - test_risk
        
        best_risk = min(best_risk, test_risk)
        a_distance = 2 * (1 - 2 * best_risk)
        
    return a_distance


def A_distance_NN(source_feature: torch.Tensor, target_feature: torch.Tensor,
               device=torch.device("cuda"), progress=True, training_epochs=10):
    """
    Calculate the :math:`\mathcal{A}`-distance, which is a measure for distribution discrepancy.
    The definition is :math:`dist_\mathcal{A} = 2 (1-2\epsilon)`, where :math:`\epsilon` is the
    test error of a classifier trained to discriminate the source from the target.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        device (torch.device)
        progress (bool): if True, displays a the progress of training A-Net
        training_epochs (int): the number of epochs when training the classifier
    Returns:
        :math:`\mathcal{A}`-distance
    """
    source_label = torch.ones((source_feature.shape[0], 1))
    target_label = torch.zeros((target_feature.shape[0], 1))
    feature = torch.cat([source_feature, target_feature], dim=0)
    label = torch.cat([source_label, target_label], dim=0)
    
    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    
    anet = ANet(feature.shape[1]).to(device)
    from torch.optim import SGD
    import torch.nn.functional as F
    optimizer = SGD(anet.parameters(), lr=0.01)
    a_distance = 2.0
    for epoch in range(training_epochs):
        anet.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            anet.zero_grad()
            y = anet(x)
            loss = F.binary_cross_entropy(y, label)
            loss.backward()
            optimizer.step()
        
        anet.eval()
        meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y = anet(x)
                acc = binary_accuracy(y, label)
                meter.update(acc, x.shape[0])
        error = 1 - meter.avg / 100
        if error>0.5:
            error = 1-error
        a_distance = 2 * (1 - 2 * error)
        if progress:
            print("epoch {} accuracy: {} A-dist: {}".format(epoch, meter.avg, a_distance))
    
    return a_distance


class ANet(nn.Module):
    def __init__(self, in_feature):
        super(ANet, self).__init__()
        self.layer = nn.Linear(in_feature, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x


class AverageMeter(object):
    r"""Computes and stores the average and current value.
    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    
def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct