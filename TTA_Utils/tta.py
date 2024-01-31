#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class TTA(nn.Module):

    def __init__(self, model, optimizer, device=torch.device('cuda'), alpha=2, temp=0.1,
                filter_K=20, neighbor_K=5, optim_steps=1, model_episodic=False,
                contrastive_=1.0, NA_=1.0, **kwargs):
        super().__init__()
        
        # hyperparameters
        self.device = device
        self.NA_ = NA_
        self.contrastive_ = contrastive_
        self.alpha = alpha
        self.model = model
        self.temp = temp
        self.filter_K = filter_K
        self.steps = optim_steps
        self.neighbor_K = neighbor_K
        self.episodic = model_episodic
        assert optim_steps > 0, "TTA requires >= 1 step(s) to forward and update"
        
        # models
        self.optimizer = optimizer
        self.classifier = model[-1]
        self.featurizer = nn.Sequential(model[0], model[1])
        
        # memory_bank initialization from the pre-trained source model
        warmup_supports = self.classifier.fc.weight.data.detach()
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.warmup_scores = F.softmax(warmup_prob, dim=1)
        # memory_bank
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.scores = self.warmup_scores.data
        
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.feature_all = None
        self.label_all = None
        self.index_certain_all = None

    def forward(self, x, y, file_name=None):
        self.supports_old = self.supports.clone()
        self.scores_old = self.scores.clone()
        self.labels_old = self.labels.clone()

        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            logits, acc_certain, acc_uncertain, loss, scores, labels, supports = self.forward_and_adapt(x, y)


        self.supports = supports
        self.scores = scores
        self.labels = labels
        # slim memory bank for TTA procedure over the next mini-batch of the online unlabeled target data
        self.supports_update()
        from Lib.train_utils import cal_acc_P
        acc_t_tar = cal_acc_P(logits, y)[0]
        return acc_t_tar, acc_certain, acc_uncertain, loss

    # @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, y):
        """Forward and adapt model on current mini-batch of the online unlabeled target data.
        """
        # forward

        features = self.featurizer(x)
        logits = self.classifier(features)
        yhat = F.one_hot(logits.argmax(1), num_classes=self.num_classes).float()
        
        # prediction confidence
        prob_predictions = F.softmax(logits, 1)
        prob, pred = prob_predictions.max(1)
        prob_mean = prob.mean()
        y_certain = (prob >= prob_mean)
        y_uncertain = ~y_certain
        
        # spectral entropy
        from scipy.stats import entropy
        e = x ** 2
        e = e / e.sum(dim=2, keepdim=True)
        e = entropy(e.cpu(), axis=2)
        e_mean = e.mean()
        e_certain = torch.flatten(torch.from_numpy(e < e_mean)).to(self.device)
        e_uncertain = ~e_certain

        # perform dynamic data division to divide each mini-bath of the online unlabeled target data
        # into the certain-aware and uncertain-aware sets for fine-grained online target model adaptation.
        index_certain = e_certain & y_certain
        index_uncertain = ~index_certain
        self.index_certain = index_certain
        self.index_uncertain = index_uncertain

        acc_certain = (pred[index_certain] == y[index_certain]).float().mean() * 100
        acc_uncertain = (pred[~index_certain] == y[~index_certain]).float().mean() * 100

        with torch.no_grad():
            # update memory bank
            supports = torch.cat([self.supports_old, features[index_certain, :]])
            labels = torch.cat([self.labels_old, yhat[index_certain, :]])
            scores = torch.cat([self.scores_old, prob_predictions[index_certain, :]])
            # calculate the class-wise prototypes from the reliable memory bank
            prototypes = (F.normalize(supports, dim=1).T @ (labels)) / torch.sum(labels, dim=0, keepdim=True)
            self.prototypes = F.normalize(prototypes, dim=0).detach()  # D*K

            # supports N*D
            # labels N*K

        # smooth entropy minimization
        from TTA_Utils.utils import entropy, Tsallis_entropy, Div_loss, Balance_P
        prob_predictions_b = Balance_P(prob_predictions, index_certain)
        loss1 = (Tsallis_entropy(prob_predictions_b, alpha=self.alpha, reduction='mean')- Div_loss(prob_predictions_b))
        
        loss2 = 0
        loss3 = 0
        features_uncertain = features[index_uncertain, :]
        features_certain = features[index_certain, :]
        probs_uncertain = prob_predictions[index_uncertain, :]
        probs_certain = prob_predictions[index_certain, :]
        
        postive_probs = torch.cat([self.scores_old, probs_certain]).detach()
        postive_features = torch.cat([self.supports_old, features_certain]).detach()
        
        
        if self.contrastive_ > 0:
            # Prototypical CL
            from TTA_Utils.utils import PCL
            loss2 = self.contrastive_ * PCL(features_certain=features_certain, features_uncertain=features_uncertain,
                                        prototypes=self.prototypes, labels=pred[index_certain], temp=1)

        if self.NA_ > 0:
            # neighborhood CL
            from TTA_Utils.utils import NCL
            loss3 = self.NA_ * NCL(features_uncertain=features_uncertain, supports=postive_features,scores=postive_probs,
                                neighbor_K=self.neighbor_K, probs_uncertain=probs_uncertain, temp=self.temp,)

        loss = loss1 + loss2 + loss3

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return logits, acc_certain, acc_uncertain, loss, scores, labels, supports

    def supports_update(self):
        scores = self.scores
        scores_max = self.scores.max(1)[0]
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(scores))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(scores)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(scores_max[y_hat == i], descending=True)
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.scores = self.scores[indices]

        return self.supports, self.labels

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


def collect_params_BN(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    
    BN optimization
    
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def collect_params_all(model):

    names = []
    params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)

    return params, names


def collect_params_fea(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    feature_ = nn.Sequential(model[0], model[1])
    names = []
    params = []
    for n, p in feature_.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)

    return params, names



def collect_params_cls(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    cls = model[-1]
    names = []
    params = []
    for n, p in cls.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model_all(model):
    """Configure model for TTA."""
    model.train()
    model.requires_grad_(True)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # local BN estimation
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, nn.Dropout):
            m.eval()
            # shutdown Dropout
    return model

def configure_model_BN(model):
    """Configure model for TTA"""

    model.train()
    # disable grad, to (re-)enable only with BN layers
    model.requires_grad_(False)
    # configure BN layers for optimization: enable grad + force local batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, nn.Dropout):
            m.eval()
            # shutdown Dropout
    return model

