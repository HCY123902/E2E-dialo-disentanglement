from __future__ import print_function
from cProfile import label
from enum import Enum
from turtle import pos

import torch
import torch.nn as nn

import constant

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import utils
import kmeans_pytorch

class lu(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, print_detail=False):
        super(lu, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.print_detail=print_detail
        self.log_softmax = torch.nn.LogSoftmax(dim=0)
        self.device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))

    def forward(self, features, dialogue_lengths, labels, pos_mask, sample_mask):
        result = torch.zeros(features.shape[0], requires_grad=True).to(self.device)

        loss = None

        for i, dialogue in enumerate(features):
            # Discard padded utterances
            dialogue = dialogue[:dialogue_lengths[i], :]
            dialogue_labels = labels[i, :dialogue_lengths[i]]
            dialogue_labels = dialogue_labels.contiguous().view(-1)

            resized_features = features.reshape(-1, features.size(2)).contiguous()

            contrast_feature = resized_features
            anchor_feature = dialogue

            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)

            # Numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # print("logits", logits.size(), "sample_mask", sample_mask.size())
            exp_logits = torch.exp(logits) * sample_mask[i][:dialogue_lengths[i]]
            numerator_logits = (logits * pos_mask[i][:dialogue_lengths[i]]).sum(1)
            
            denominator_logits = torch.log(exp_logits.sum(1))
            
            # Multiply with number of positive samples
            denominator_logits = denominator_logits * (pos_mask[i][:dialogue_lengths[i]].sum(1))

            loss = - (self.temperature / self.base_temperature) * (numerator_logits - denominator_logits)
            
            # Average with number of positive samples
            loss = loss / (pos_mask[i][:dialogue_lengths[i]].sum(1))

            if torch.any(loss.isnan()):
                print("NCE containing nan", loss)
                print("Corresponding NCE dialogue containing nan", dialogue)
                # assert ~torch.any(loss.isnan())
                
                loss = loss[~loss.isnan()]
            
            loss = loss.mean()

            result[i] = loss
    
        result = result[~result.isnan()]
        
        if self.print_detail:
            print("NCE result", result)
            print("NCE result mean", result.mean())

        return result.mean()


class ls(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, print_detail=False):
        super(ls, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.print_detail=print_detail
        self.device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))

    def forward(self, features, dialogue_lengths, labels=None, mask=None):
        result = torch.zeros(features.shape[0], requires_grad=True).to(self.device)

        for i, dialogue in enumerate(features):
            # Discard padded utterances  
            dialogue = dialogue[:dialogue_lengths[i], :] 
            dialogue_labels = labels[i, :dialogue_lengths[i]]
            label_range = int(dialogue_labels.max().item()) + 1
            dialogue_labels = dialogue_labels.contiguous().view(-1, 1)
            assert dialogue_labels.shape[0] == dialogue.shape[0]
            prototype_mask = torch.LongTensor(1, label_range).to(self.device)
            prototype_mask[0] = torch.LongTensor(range(label_range)).to(self.device)
            mask = torch.eq(dialogue_labels, prototype_mask).float().to(self.device)

            # state_number, hidden size
            prototypes = torch.Tensor(label_range, features.shape[2]).to(self.device)
            for k in range(label_range):
                dialogue_label_mask = (dialogue_labels[:, 0] == k).nonzero(as_tuple=True)[0]
                session = dialogue[dialogue_label_mask, :]
                prototypes[k, :] = session.mean(0)
            
            contrast_feature = prototypes
            anchor_feature = dialogue

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)

            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            # compute log_prob
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))            
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            
            if torch.any(loss.isnan()):
                print("Prototype containing nan", loss)
                print("Corresponding prototype dialogue containing nan", dialogue)
                # assert ~torch.any(loss.isnan())
                
                loss = loss[~loss.isnan()]
            
            loss = loss.view(1, dialogue_lengths[i]).mean()

            result[i] = loss
        
        result = result[~result.isnan()]
        
        if self.print_detail:
            print("Prototype result", result)
            print("Prototype result mean", result.mean())

        return result.mean()

class lm_lk(nn.Module):
    def __init__(self, print_detail=False, Kmeans_metric='silhouette'):
        super(lm_lk, self).__init__()
        self.print_detail=print_detail
        self.Kmeans_metric = Kmeans_metric
        self.device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))

    def forward(self, features, dialogue_lengths, labels=None, mask=None, k_prob=None):
        
        result = torch.zeros(features.shape[0]).to(self.device)

        for i, dialogue in enumerate(features):
            # Discard padded utterances  
            dialogue = dialogue[:dialogue_lengths[i], :]       
            dialogue_labels = labels[i, :dialogue_lengths[i]]
            label_range = int(dialogue_labels.max().item()) + 1
            dialogue_labels = dialogue_labels.contiguous().view(-1, 1)
            assert dialogue_labels.shape[0] == dialogue.shape[0]
            prototype_mask = torch.LongTensor(1, label_range).to(self.device)
            prototype_mask[0] = torch.LongTensor(range(label_range)).to(self.device)
            mask = torch.eq(dialogue_labels, prototype_mask).float().to(self.device)
            # print("Checkpoint 4 mask", mask)

            # state_number, hidden size
            prototypes = torch.Tensor(label_range, features.shape[2]).to(self.device)
            for k in range(label_range):
                
                dialogue_label_mask = (dialogue_labels[:, 0] == k).nonzero(as_tuple=True)[0]
                session = dialogue[dialogue_label_mask, :]
                # print("Checkpoint 4 dialogue_label_mask", dialogue_label_mask)
                prototypes[k, :] = session.mean(0)


            dialogue_cpu = dialogue.cpu().detach().numpy()
            k_val =  (torch.argmax(k_prob[i, :dialogue_lengths[i]]) + 1).item()
            k_means = KMeans(n_clusters=k_val, random_state=0).fit(dialogue_cpu)
            prototypes_numpy = prototypes.cpu().detach().numpy()
            squared_distance = np.array([[np.linalg.norm(prototype-center) for center in k_means.cluster_centers_] for prototype in prototypes_numpy])
            row_ind, col_ind = linear_sum_assignment(squared_distance)
            cnt = len(row_ind)
            loss = torch.tensor(0).to(self.device)

            for (r, c) in zip(row_ind, col_ind):
                loss = loss + torch.dist(prototypes[r], torch.tensor(k_means.cluster_centers_[c]).to(self.device)).to(self.device)
            # Average distance between prototype and matched cluster center
            loss = loss / cnt
            loss = 0.5 * loss + 0.5 * (-torch.log(k_prob[i, label_range - 1]))
            
            if torch.any(loss.isnan()):
                print("Matching containing nan", loss)
                print("Corresponding matching dialogue containing nan", dialogue)
                # assert ~torch.any(loss.isnan())
                
                loss = loss[~loss.isnan()]
            result[i] = loss
        
        if self.print_detail:
            print("Matching result", result)
            print("Matching result mean", result.mean())

        return result.mean()

class lsp(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, print_detail=False):
        super(lsp, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.print_detail=print_detail
        self.device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))

    def forward(self, features, dialogue_lengths, labels=None, mask=None):
        result = torch.zeros(features.shape[0], requires_grad=True).to(self.device)

        for i, dialogue in enumerate(features):
            for m in range(1, min(dialogue_lengths[i] - 1, constant.state_num) + 1):
                # Discard padded utterances  
                dialogue = dialogue[:dialogue_lengths[i], :]
                # print("Checkpoint 1 dialgoue", dialgoue)

                prototypes = torch.Tensor(m, features.shape[2]).to(self.device)
                if m > 1:
                    cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(X=dialogue, num_clusters=m, distance='euclidean', device=self.device, tqdm_flag=False)
                else:
                    cluster_ids_x = torch.zeros(dialogue.size(0))
                    cluster_centers = dialogue.mean(dim=0, keepdim=True)
                prototypes = cluster_centers.to(self.device)
                
                dialogue_labels = cluster_ids_x.to(self.device)
                label_range = int(prototypes.size(0))
                dialogue_labels = dialogue_labels.contiguous().view(-1, 1)
                assert dialogue_labels.shape[0] == dialogue.shape[0]
                prototype_mask = torch.LongTensor(1, label_range).to(self.device)
                prototype_mask[0] = torch.LongTensor(range(label_range)).to(self.device)

                mask = torch.eq(dialogue_labels, prototype_mask).float().to(self.device)
                
                contrast_feature = prototypes
                anchor_feature = dialogue
                anchor_dot_contrast = torch.div(
                    torch.matmul(anchor_feature, contrast_feature.T),
                    self.temperature)

                # for numerical stability
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
                
                mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                
                if torch.any(loss.isnan()):
                    print("Prototype containing nan", loss)
                    print("Corresponding prototype dialogue containing nan", dialogue)
                    # assert ~torch.any(loss.isnan())
                    
                    loss = loss[~loss.isnan()]
                
                loss = loss.view(1, dialogue_lengths[i]).mean()

                result[i] = result[i] + loss
            result[i] = result[i] / constant.state_num

        result = result[~result.isnan()]
        
        if self.print_detail:
            print("Prototype result", result)
            print("Prototype result mean", result.mean())

        return result.mean()
        