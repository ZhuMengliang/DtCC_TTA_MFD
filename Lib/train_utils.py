#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

import numpy as np
from collections import deque
import torch,os,random
import torch.nn as nn
import torch.nn.functional as F


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    # CADN
    # \frac{1-exp(-\delta*p)}{1+exp(-\delta*p)},\delta=10,p is the training progress changing from 0 to 1,
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def cal_acc(loader, model, device=torch.device("cuda"), analysis=False):

    model.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1].to(device)
            inputs = inputs.to(device)
            features = model[1](model[0](inputs))
            outputs = model[2](features)
            if start_test:
                all_output = outputs.float().to(device)
                all_label = labels.float().to(device)
                all_feature = features.float().to(device)
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().to(device)), 0)
                all_label = torch.cat((all_label, labels.float().to(device)), 0)
                all_feature = torch.cat((all_feature, features.float().to(device)), 0)
    
    prob = nn.Softmax(dim=1)(all_output).to(device)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.mean((torch.squeeze(predict).float() == all_label).float()).item()
    
    if analysis:
        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(all_label.cpu(), torch.squeeze(predict.cpu()).float())
        matrix = matrix[np.unique(all_label.cpu()).astype(int), :]
        # 混淆矩阵 行代表真实标签，列代表预测标签
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Times New Roman']
        rcParams['font.size'] = 15
        import seaborn as sns
        ax = sns.heatmap(matrix / matrix.sum(axis=0,keepdims=True), annot=True, fmt='.2%', cmap='Greens')
        ax.title.set_text("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        filename = 'Confusion_matrix.jpg'
        plt.savefig(filename, format='jpg', dpi=480, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        
        all_acc = matrix.diagonal() / matrix.sum(axis=0) * 100  # precision for each class
        accuracy = matrix.diagonal().sum() / matrix.sum()
        aa = [str(np.round(i, 2)) for i in all_acc]
        acc_list = ' '.join(aa)
        all_recall = matrix.diagonal() / matrix.sum(axis=1) * 100
        aa = [str(np.round(i, 2)) for i in all_recall]
        recall_list = ' '.join(aa)
        print("***********************************")
        print("acc for each class is ", acc_list)
        print("recall for each class is", recall_list)
        print(f"the ovaerall accuracy is {accuracy * 100:.2f}")
        print("the confusion matrix is:\n ", matrix)
        print("***********************************")
        
    return accuracy * 100, prob.detach().clone(), F.normalize(all_feature, p=2, dim=1), all_label, predict
    '''
    if analysis:
        num_class = len(torch.unique(all_label))
        entropy = (torch.sum(-prob*torch.log(prob),dim=1)/torch.log(torch.tensor(num_class))).cpu()
        std = entropy.var().sqrt()
        m = entropy.mean()
        import matplotlib.pyplot as plt
        # plt.hist(entropy.cpu().numpy(), bins=np.arange(0, 1.01, 0.1), edgecolor='k')
        # plt.xticks(np.arange(0, 1.01, 0.1))
        # plt.show()

        index_list = [((entropy>i)&(entropy<=i+0.1)) for i in np.arange(0,1,0.1)]
        acc_list = [(predict[index_]==all_label[index_]).float().sum() for index_ in index_list]
        entropy_list = [index.sum().cpu().numpy() for index in index_list]

        index_=np.arange(0,1,0.1)
        width=0.05
        plt.bar(index_+width/2,entropy_list,width=0.05,edgecolor='k',label='en_num')
        plt.bar(index_+1.5*width, acc_list,width=0.05, edgecolor='k',label='acc_num')
        plt.axvline(x=m,color='r',linestyle='--',label='mean')
        plt.axvline(x=m+std, color='b', linestyle='--', label='mean+2std')
        plt.axvline(x=m-std, color='b', linestyle='--', label='mean-2std')
        plt.legend()
        plt.xticks(index_)
        plt.show()
    '''


def cal_acc_P(outputs_P,Y):
    _, predict = torch.max(outputs_P, 1)
    accuracy = torch.mean((torch.squeeze(predict).float() == Y).float()).item()
    return accuracy*100, predict


def cal_acc_old(loader, model, device=torch.device("cuda"),analysis=False,
            center=False, balanced=True, index=None):
    centroid = None
    model.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1].to(device)
            inputs = inputs.to(device)
            features = model[1](model[0](inputs))
            outputs = model[2](features)
            if start_test:
                all_output = outputs.float().to(device)
                all_label = labels.float().to(device)
                all_feature = features.float().to(device)
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().to(device)), 0)
                all_label = torch.cat((all_label, labels.float().to(device)), 0)
                all_feature = torch.cat((all_feature, features.float().to(device)), 0)
    
    
    prob = nn.Softmax(dim=1)(all_output).to(device)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.mean((torch.squeeze(predict).float() == all_label).float()).item()
    
    if center:
        if index == None:
            index = torch.ones_like(predict,dtype=torch.bool)
        all_feature = all_feature / (torch.norm(all_feature, p=2, dim=1, keepdim=True) + 1e-8)
        
        if balanced:
            num_classes = all_output.size()[1]
            centroid = torch.zeros(num_classes, all_feature.size()[1])
            rate = 1.5
            # topk_k = int(max(all_feature.size()[0] // (num_classes * rate), 1)) # 根据给定的比例划分固定的类别数目
            # # topk_k = int(torch.unique(predict.cpu(), return_counts=True)[1].min()) #按照每次分类中类别最小的数目确定
            #
            topk_k = int(max(torch.unique(predict.cpu(), return_counts=True)[1].min(),
                         all_feature.size()[0] // (num_classes * rate))) # 自适应选取，取最大值。
            
            for cls_index in range(num_classes):
                feat_index = torch.topk(all_output[:, cls_index], topk_k)[1]
                feat_cls = all_feature[feat_index, :]
                centroid[cls_index, :] = feat_cls.mean(dim=0) + 1e-8
            
            centroid = ((centroid / torch.norm(centroid, p=2, dim=1, keepdim=True)) + 1e-8).to(device)
            dis = torch.mm(all_feature, centroid.t().to(device))
            _, predict_p = torch.max(dis, dim=1)
            acc_pro = (torch.squeeze(predict_p).float().to(device) == all_label.to(device)).float().mean() * 100
            print(f"the Acc_neural/prototype is {accuracy * 100:.2f}/{acc_pro:.2f}")
            
        else:
            all_fea = all_feature.float().cpu().numpy()
            aff = np.eye(all_output.size(1))[predict[index].cpu()]
            centroid = aff.transpose().dot(all_fea[index.cpu(), :])
            centroid = centroid / (1e-8 + aff.sum(axis=0)[:, None])
            centroid = torch.from_numpy(centroid).float().to(device)
            
            centroid = (centroid / torch.norm(centroid, p=2, dim=1, keepdim=True)) + 1e-8
            dis = torch.mm(torch.from_numpy(all_fea).float().to(device), centroid.t())
            _, predict_p = torch.max(dis, dim=1)
            acc_pro = (torch.squeeze(predict_p).float().to(device) == all_label.to(device)).float().mean() * 100
            print(f"the Acc_neural/prototype is {accuracy * 100:.2f}/{acc_pro:.2f}")
            
    if analysis:
        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(all_label.cpu(), torch.squeeze(predict.cpu()).float())
        matrix = matrix[np.unique(all_label.cpu()).astype(int), :]
        all_acc = matrix.diagonal() / matrix.sum(axis=1) * 100 # precision for each class
        accuracy = matrix.diagonal().sum() / matrix.sum()
        aa = [str(np.round(i, 2)) for i in all_acc]
        acc_list = ' '.join(aa)
        
        all_recall = matrix.diagonal() / matrix.sum(axis=0) * 100
        aa = [str(np.round(i, 2)) for i in all_recall]
        recall_list = ' '.join(aa)
        print("***********************************")
        print("acc for each class is ", acc_list)
        print("recall for each class is", recall_list)
        print(f"the ovaerall accuracy is {accuracy*100:.2f}")
        print("the confusion matrix is:\n ", matrix)
        print("***********************************")
        
        
        
    '''
    if analysis:
        num_class = len(torch.unique(all_label))
        entropy = (torch.sum(-prob*torch.log(prob),dim=1)/torch.log(torch.tensor(num_class))).cpu()
        std = entropy.var().sqrt()
        m = entropy.mean()
        import matplotlib.pyplot as plt
        # plt.hist(entropy.cpu().numpy(), bins=np.arange(0, 1.01, 0.1), edgecolor='k')
        # plt.xticks(np.arange(0, 1.01, 0.1))
        # plt.show()

        index_list = [((entropy>i)&(entropy<=i+0.1)) for i in np.arange(0,1,0.1)]
        acc_list = [(predict[index_]==all_label[index_]).float().sum() for index_ in index_list]
        entropy_list = [index.sum().cpu().numpy() for index in index_list]
        
        index_=np.arange(0,1,0.1)
        width=0.05
        plt.bar(index_+width/2,entropy_list,width=0.05,edgecolor='k',label='en_num')
        plt.bar(index_+1.5*width, acc_list,width=0.05, edgecolor='k',label='acc_num')
        plt.axvline(x=m,color='r',linestyle='--',label='mean')
        plt.axvline(x=m+std, color='b', linestyle='--', label='mean+2std')
        plt.axvline(x=m-std, color='b', linestyle='--', label='mean-2std')
        plt.legend()
        plt.xticks(index_)
        plt.show()
    '''
    return accuracy * 100, prob.detach().clone(),  F.normalize(all_feature, p=2, dim=1), all_label, centroid, predict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table
    
    def update(self, val, n=1, history=True, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)  # 记录的是先前(step-1)个 epoch+当前epoch的平均值
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)
    
    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]
    
    def __len__(self):
        return self.count
    
    
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def tsne_plot_features(features, labels, index_certain, filename='tsne.jpg',
                       norm=True, perplexity=30):
    features = features.cpu().detach().numpy()
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

    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500 # 分辨率
    plt.rcParams['figure.figsize'] = (5.0, 5.0)

    #这里的perplexity是一个关键因素
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=42)
    features_tsne = tsne.fit_transform(features)
    import pandas as pd
    df = pd.DataFrame(features_tsne[:, 0], columns=['dim0'])
    df['dim1'] = features_tsne[:, 1]
    df['Category'] = labels.cpu()
    df['certain'] = index_certain.cpu().detach().float()
    # 0 对应的是圆圈，代表不确定的
    # 1 对应的是 ×，代表确定的
    df['size'] = 10 * np.ones(features_tsne.shape[0])
    axe = sns.scatterplot(
        data=df,
        x="dim0", y="dim1",
        hue='Category',
        style="certain",
        palette=sns.color_palette("hls", torch.unique(labels).size()[0]),
        size='size'
    )
    handles, labels = axe.get_legend_handles_labels()
    handles = handles[0:13]+handles[15:18]
    axe.legend(handles, ['Category', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                         'Set', 'Uncertain', 'Certain'],
               loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # axe.legend_.remove()  # 去掉图例
    # axe.spines['top'].set_visible(False)
    # axe.spines['right'].set_visible(False)
    # axe.spines['bottom'].set_visible(False)
    # axe.spines['left'].set_visible(False)
    # plt.xticks([])  # 去掉刻度
    # plt.yticks([])
    # axe.set(xticklabels=[])
    # axe.set(xlabel=None)
    # axe.set(yticklabels=[])
    # axe.set(ylabel=None)
    plt.savefig(filename, format='jpg', dpi=500,bbox_inches = 'tight')
    #plt.show()
    plt.close()
    plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
    plt.clf()  # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
    plt.close()  # 关闭 window，如果没有指定，则指当前 window。
    del plt  # 这个地方要把plt这个变量给删掉，不然下面主函数调用的时候，一旦使用了plt.show()，也会把这里的图给显示出来。



def tsne_plot_features_all(features, labels, index_certain, filename='tsne.jpg',
                       norm=True, perplexity=30):
    features = features.cpu().detach().numpy()
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

    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500 # 分辨率
    plt.rcParams['figure.figsize'] = (5.0, 5.0)

    #这里的perplexity是一个关键因素
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=42)
    features_tsne = tsne.fit_transform(features)
    import pandas as pd
    df = pd.DataFrame(features_tsne[:, 0], columns=['dim0'])
    df['dim1'] = features_tsne[:, 1]
    df['Category'] = labels.cpu()
    df['certain'] = index_certain.cpu().detach().float()
    # 0 对应的是圆圈，代表不确定的
    # 1 对应的是 ×，代表确定的
    df['size'] = 10 * np.ones(features_tsne.shape[0])
    label_num = torch.unique(labels).size()[0]
    axe = sns.scatterplot(
        data=df,
        x="dim0", y="dim1",
        hue='Category',
        style="certain",
        palette=sns.color_palette("hls", label_num),
        size='size'
    )
    handles, labels_ = axe.get_legend_handles_labels()
    handles = handles[0:label_num+2]+handles[label_num+4:label_num+7]

    C_str = [str(i) for i in range(label_num)]
    C_str = ['Category'] + C_str + ['Set', 'Uncertain', 'Certain']
    axe.legend(handles, C_str, loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # axe.legend_.remove()  # 去掉图例
    # axe.spines['top'].set_visible(False)
    # axe.spines['right'].set_visible(False)
    # axe.spines['bottom'].set_visible(False)
    # axe.spines['left'].set_visible(False)
    # plt.xticks([])  # 去掉刻度
    # plt.yticks([])
    # axe.set(xticklabels=[])
    # axe.set(xlabel=None)
    # axe.set(yticklabels=[])
    # axe.set(ylabel=None)
    plt.savefig(filename, format='jpg', dpi=500,bbox_inches = 'tight')
    #plt.show()
    plt.close()
    plt.cla()  # 清除axes，即当前 figure 中的活动的axes，但其他axes保持不变。
    plt.clf()  # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
    plt.close()  # 关闭 window，如果没有指定，则指当前 window。
    del plt