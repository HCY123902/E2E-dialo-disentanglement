from __future__ import print_function

import torch
import torch.nn as nn

import constant

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

class SupConLossNCE(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossNCE, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, dialogue_lengths, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, max_dialogue_length, hidden_size].
            labels: ground truth of shape [bsz, max_dialogue_length].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))
        result = torch.zeros(features.shape[0]).to(device)

        for i, dialogue in enumerate(features):
            # Discard padded utterances
            dialogue = dialogue[:dialogue_lengths[i], :]
            
#             print("Checkpoint alpha", dialgoue)

            dialogue_labels = labels[i, :dialogue_lengths[i]]
            dialogue_labels = dialogue_labels.contiguous().view(-1, 1)
            assert dialogue_labels.shape[0] == dialogue.shape[0]
            # if dialogue_labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(dialogue_labels, dialogue_labels.T).float().to(device)

            contrast_feature = dialogue
            anchor_feature = dialogue
            
            # print(contrast_feature.shape, anchor_feature.shape)

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
#             print("Checkpoint a", anchor_feature)
#             print("Checkpoint b", contrast_feature)
#             print("Checkpoint c", anchor_dot_contrast)
            
            
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            
#             print("Checkpoint d", logits)

            # tile mask
            # mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(dialogue_lengths[i]).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask
            
            empty_mask = ~torch.all(mask == 0,dim=1)
            
            mask = mask[empty_mask]
            logits_mask = logits_mask[empty_mask]
            logits = logits[empty_mask]
            
            
            
#             print("Checkpoint 1 mask", mask)

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            
#             print("Checkpoint 2 exp_logits", exp_logits)
            
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            
#             print("Checkpoint 3 log_prob", log_prob)

            # compute mean of log-likelihood over positive
#             print("Checkpoint 4 log_prob shape", log_prob.shape)
#             print("Checkpoint 4 mask * log_prob", mask * log_prob)
#             print("Checkpoint 4 (mask * log_prob).sum(1)", (mask * log_prob).sum(1))

            
            
            
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            
            if torch.any(loss.isnan()):
                print("NCE containing nan", loss)
                print("Corresponding NCE dialogue containing nan", dialogue)
                # assert ~torch.any(loss.isnan())
                
                loss = loss[~loss.isnan()]
            
#             print("Checkpoint 5 loss", loss)
            
            loss = loss.view(1, -1).mean()
            
#             print("Checkpoint 6 loss", loss)

            result[i] = loss
    
        result = result[~result.isnan()]
            
        print("NCE result", result)
        print("NCE result mean", result.mean())

        return result.mean()



        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        # batch_size = features.shape[0]
        # if labels is not None and mask is not None:
        #     raise ValueError('Cannot define both `labels` and `mask`')
        # elif labels is None and mask is None:
        #     mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # elif labels is not None:
        #     labels = labels.contiguous().view(-1, 1)
        #     if labels.shape[0] != batch_size:
        #         raise ValueError('Num of labels does not match num of features')
        #     mask = torch.eq(labels, labels.T).float().to(device)
        # else:
        #     mask = mask.float().to(device)

        # contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # if self.contrast_mode == 'one':
        #     anchor_feature = features[:, 0]
        #     anchor_count = 1
        # elif self.contrast_mode == 'all':
        #     anchor_feature = contrast_feature
        #     anchor_count = contrast_count
        # else:
        #     raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # # compute logits
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

        # # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        # mask = mask * logits_mask

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()

        # return loss

class SupConLossPrototype(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossPrototype, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, dialogue_lengths, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, max_dialogue_length, hidden_size].
            labels: ground truth of shape [bsz, max_dialogue_length].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))
        result = torch.zeros(features.shape[0]).to(device)

        for i, dialogue in enumerate(features):
            # Discard padded utterances  
            dialogue = dialogue[:dialogue_lengths[i], :]
            # print("Checkpoint 1 dialgoue", dialgoue)
            
            dialogue_labels = labels[i, :dialogue_lengths[i]]
            
            # print("Checkpoint 2 dialogue_labels", dialogue_labels)
            
            label_range = int(dialogue_labels.max().item()) + 1
            
            # print("Checkpoint 3 largest_label", label_range)
            
            dialogue_labels = dialogue_labels.contiguous().view(-1, 1)
            assert dialogue_labels.shape[0] == dialogue.shape[0]
            # if dialogue_labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')

            prototype_mask = torch.LongTensor(1, label_range).to(device)
            prototype_mask[0] = torch.LongTensor(range(label_range)).to(device)
            
            # print("Checkpoint 3 prototype_mask", prototype_mask)

            mask = torch.eq(dialogue_labels, prototype_mask).float().to(device)
            
            # print("Checkpoint 4 mask", mask)

            # state_number, hidden size
            prototypes = torch.Tensor(label_range, features.shape[2]).to(device)
            for k in range(label_range):
                
                dialogue_label_mask = (dialogue_labels[:, 0] == k).nonzero(as_tuple=True)[0]
                # print(dialogue_label_mask.shape)
                # session = dialgoue[dialogue_labels[i] == k, :]
                # session = dialgoue[dialogue_label_mask]
                session = dialogue[dialogue_label_mask, :]
                # print("Checkpoint 4 dialogue_label_mask", dialogue_label_mask)
                # print("Checkpoint 4 session", session)
                # print("Checkpoint 4 session.mean(0)", session.mean(0))
                prototypes[k, :] = session.mean(0)

            # Include the entire conversation when calculating the session prototype regardless of the anchor position since the entire conversation will be available in the response ranking task

            # print("Checkpoint 5 prototypes", prototypes)
            
            contrast_feature = prototypes
            anchor_feature = dialogue

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            # for numerical stability
            
            # print("Checkpoint 6 anchor_dot_contrast", anchor_dot_contrast)
            
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            
            # print("Checkpoint 7 logits", logits)

            # tile mask
            # mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            # logits_mask = torch.scatter(
            #     torch.ones_like(mask),
            #     1,
            #     torch.arange(dialogue_lengths[i]).view(-1, 1).to(device),
            #     0
            # )
            # mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits)
            
            # print("Checkpoint 8 exp_logits", exp_logits)
            
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            
            # print("Checkpoint 9 log_prob", log_prob)

            # compute mean of log-likelihood over positive
            # print(mask.shape)
            # print(log_prob.shape)
            
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            
            # print("Checkpoint 10 mean_log_prob_pos", mean_log_prob_pos)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            
            if torch.any(loss.isnan()):
                print("Prototype containing nan", loss)
                print("Corresponding prototype dialogue containing nan", dialogue)
                # assert ~torch.any(loss.isnan())
                
                loss = loss[~loss.isnan()]
            
            loss = loss.view(1, dialogue_lengths[i]).mean()

            result[i] = loss
        
        result = result[~result.isnan()]
            
        print("Prototype result", result)
        print("Prototype result mean", result.mean())

        return result.mean()

class PrototypeKmeansDivergence(nn.Module):
    def __init__(self):
        super(PrototypeKmeansDivergence, self).__init__()
        self.M = 0
        self.N = 0
        self.Map = torch.zeros((constant.state_num, constant.state_num), dtype=int)
        self.p = torch.zeros(constant.state_num, dtype=int)
        self.vis = torch.zeros(constant.state_num, dtype=int)

    def forward(self, features, dialogue_lengths, labels=None, mask=None):
        device = (torch.device('cuda')
                  if torch.cuda.is_available()
                  else torch.device('cpu'))
        result = torch.zeros(features.shape[0]).to(device)

        for i, dialogue in enumerate(features):
            # Discard padded utterances  
            dialogue = dialogue[:dialogue_lengths[i], :]
            # print("Checkpoint 1 dialgoue", dialgoue)
            
            dialogue_labels = labels[i, :dialogue_lengths[i]]
            
            # print("Checkpoint 2 dialogue_labels", dialogue_labels)
            
            label_range = int(dialogue_labels.max().item()) + 1
            
            # print("Checkpoint 3 largest_label", label_range)
            
            dialogue_labels = dialogue_labels.contiguous().view(-1, 1)
            assert dialogue_labels.shape[0] == dialogue.shape[0]
            # if dialogue_labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')

            prototype_mask = torch.LongTensor(1, label_range).to(device)
            prototype_mask[0] = torch.LongTensor(range(label_range)).to(device)
            
            # print("Checkpoint 3 prototype_mask", prototype_mask)

            mask = torch.eq(dialogue_labels, prototype_mask).float().to(device)
            
            # print("Checkpoint 4 mask", mask)

            # state_number, hidden size
            prototypes = torch.Tensor(label_range, features.shape[2]).to(device)
            for k in range(label_range):
                
                dialogue_label_mask = (dialogue_labels[:, 0] == k).nonzero(as_tuple=True)[0]
                # print(dialogue_label_mask.shape)
                # session = dialgoue[dialogue_labels[i] == k, :]
                # session = dialgoue[dialogue_label_mask]
                session = dialogue[dialogue_label_mask, :]
                # print("Checkpoint 4 dialogue_label_mask", dialogue_label_mask)
                # print("Checkpoint 4 session", session)
                # print("Checkpoint 4 session.mean(0)", session.mean(0))
                prototypes[k, :] = session.mean(0)

            # Include the entire conversation when calculating the session prototype regardless of the anchor position since the entire conversation will be available in the response ranking task

            # print("Checkpoint 5 prototypes", prototypes)

            cluster_number = int((dialogue_lengths[i] / float(constant.utterance_max_length)) * (constant.state_num))
            # print("cluster number", cluster_number)
            
            k_means = KMeans(n_clusters=cluster_number, random_state=0).fit(dialogue.numpy())
            
            squared_distance = np.array([[np.linalg.norm(prototype.numpy()-center) for center in k_means.cluster_centers_] for prototype in prototypes])

            row_ind, col_ind = linear_sum_assignment(squared_distance)
            cnt = len(row_ind)

            loss = torch.tensor(0).to(device)

            for (r, c) in zip(row_ind, col_ind):
              loss = loss + torch.dist(prototypes[r], torch.tensor(k_means.cluster_centers_[c]).to(device)).to(device)
            # Average distance between prototype and matched cluster center
            loss = loss / cnt
            
            if torch.any(loss.isnan()):
                print("Prototype containing nan", loss)
                print("Corresponding prototype dialogue containing nan", dialogue)
                # assert ~torch.any(loss.isnan())
                
                loss = loss[~loss.isnan()]
            
            result[i] = loss
            
        print("Prototype result", result)
        print("Prototype result mean", result.mean())

        return result.mean()
