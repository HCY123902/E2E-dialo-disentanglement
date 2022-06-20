from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

import sys
import os
import numpy as np 
import logging 
from tqdm import tqdm

import constant
import utils

import criterion

from sklearn.cluster import KMeans

class SupervisedTrainer(object):
    def __init__(self, args, ensemble_model, logger=None, current_time=None, loss=nn.KLDivLoss, optimizer=None):
        self.args = args
        self.ensemble_model = ensemble_model
        self.logger = logger
        self.current_time = current_time
        self.loss_func = loss(reduction='batchmean')
        self.device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))

        if self.args.train_mode == 'supervised':
            self.ls = criterion.ls(temperature=constant.temperature, base_temperature=constant.base_temperature, print_detail=args.print_detail)
            self.lm_lk = criterion.lm_lk(print_detail=args.print_detail, Kmeans_metric=args.Kmeans_metric)
        elif self.args.train_mode == 'unsupervised':
            self.lsp = criterion.lsp(temperature=constant.temperature, base_temperature=constant.base_temperature, print_detail=args.print_detail)
        self.lu = criterion.lu(temperature=constant.temperature, base_temperature=constant.base_temperature, print_detail=args.print_detail)

        if optimizer == None:
            params = list(self.ensemble_model.parameters())
            self.optimizer = optim.Adam(params, lr=constant.learning_rate)
        else:
            self.optimizer = optimizer
    
    def calculate_loss(self, input_, target, loss_mask):
        target = target[:, :input_.size(1)]
        if torch.cuda.is_available():
            target = torch.cuda.LongTensor(target)
        else:
            target = torch.LongTensor(target)
        loss = -input_.gather(2, target.unsqueeze(2)).squeeze(2)*loss_mask
        loss = torch.sum(loss)/len(input_)
        return loss

    def _train_batch(self, batch):
        batch_utterances, utterance_sequence_length, conversation_length, padded_labels, padded_speakers, pos_mask, sample_mask = batch
        attentive_repre, k_prob = self.ensemble_model(batch)
        if torch.any(attentive_repre.isnan()):
            return "skip", 0, 0, 0
        
        
        if self.args.train_mode == 'supervised':
            loss_1 = self.lu(attentive_repre, conversation_length, padded_labels, pos_mask, sample_mask)
            loss_2 = self.ls(attentive_repre, conversation_length, padded_labels)
            loss_3 = self.lm_lk(attentive_repre, conversation_length, padded_labels, k_prob=k_prob)

            loss = constant.lu_weightage * loss_1 + constant.ls_weightage * loss_2 + (1 - constant.lu_weightage - constant.ls_weightage) * loss_3

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.data.item(), len(batch_utterances)

        elif self.args.train_mode == 'unsupervised':
            loss_1 = self.lu(attentive_repre, conversation_length, padded_labels, pos_mask, sample_mask)
            loss_2 = self.lsp(attentive_repre, conversation_length, padded_labels)
            loss = constant.lu_weightage * loss_1 + (1 - constant.lu_weightage) * loss_2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.data.item(), len(batch_utterances)

    def train(self, train_loader, dev_loader):
        step_cnt = 0
        for epoch in tqdm(range(constant.epoch_num)):
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_loader)):
                step_cnt += 1
                loss, batch_size = self._train_batch(batch)
                
                # Added
                if loss == "skip":
                    print("Skip the current batch with nan value")
                    continue
                
                epoch_loss += loss
                log_msg = "Epoch : {}, batch: {}/{}, step: {}, batch loss: {}".format(
                                        epoch, i, len(train_loader), step_cnt, round(loss, 4))
                self.logger.info(log_msg)
                if step_cnt % constant.inference_step == 0:
                    purity_score, nmi_score, ari_score, shen_f_score, accuracy_k, difference_k = self.evaluate(dev_loader, step_cnt)
                    log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}, accuracy_k: {}, difference_k： {}".format(
                        round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4), round(accuracy_k, 4), round(difference_k, 4))
                    self.logger.info(log_msg)

                    model_name = os.path.join(constant.save_model_path, "model_{}".format(self.current_time), \
                                                    "step_{}.pkl".format(step_cnt))
                    log_msg = "Saving model for step {} at '{}'".format(step_cnt, model_name)
                    self.logger.info(log_msg)
                    torch.save(self.ensemble_model.state_dict(), model_name)

            log_msg = "Epoch average loss is: {}".format(round(epoch_loss/len(train_loader), 4))
            self.logger.info(log_msg)

    def init_state(self, batch_index, j, state, hidden_state_history, utterance_repre, conversation_repre, predicted_batch_label, mask):
        # state : [5, constant.hidden_size]
        if j == 0:
            one_res = self.ensemble_model.state_matrix_encoder.pooling(state[1:, :].unsqueeze(0))[0][0]
            state[0] = self.ensemble_model.state_matrix_encoder.new_state_projection(
                torch.cat([one_res, conversation_repre[batch_index][j]])
            )
        else:
            label = predicted_batch_label[-1]
            if label == 0:
                position = mask.cpu().tolist().index(-1.)
                mask[position] = 0. 
                new_output, new_hidden = self.ensemble_model.session_encoder.encoder(utterance_repre[batch_index][j-1].unsqueeze(0).unsqueeze(0))
            else:
                position = label
                # state[label]: [hidden_size]
                new_output, new_hidden = self.ensemble_model.session_encoder.encoder(utterance_repre[batch_index][j-1].unsqueeze(0).unsqueeze(0), hidden_state_history[label])
                # new_output: [1, 1, hidden_size]
            state[position] = new_output.squeeze()
            hidden_state_history[position] = new_hidden
            one_res = self.ensemble_model.state_matrix_encoder.pooling(state[1:, :].unsqueeze(0))[0][0]
            state[0] = self.ensemble_model.state_matrix_encoder.new_state_projection(
                torch.cat([one_res, conversation_repre[batch_index][j-1]])
            )
        return state, mask, hidden_state_history
    
    def predict(self, batch_index, j, state, utterance_repre, conversation_repre, mask):
        # state: [5, hidden_size]
        current_uttr_repre_concat = torch.cat((utterance_repre[batch_index][j], conversation_repre[batch_index][j]), 0)
        # [hidden_size * 2]
        current_uttr_repre = self.ensemble_model.scores_calculator.utterance_projection(current_uttr_repre_concat)
        scores = torch.matmul(current_uttr_repre, state.permute(1, 0))
        masked_scores = scores + mask*10000
        softmax_masked_scores = nn.Softmax(dim=0)(masked_scores)
        label = softmax_masked_scores.topk(2)[1].cpu().numpy()
        if label[0] == 0 and -1. not in mask:
            ret_label = label[1]
        else:
            ret_label = label[0]
        return ret_label

    def evaluate(self, test_loader, step_cnt):
        predicted_labels = []
        truth_labels = []
        count_k = 0
        correct_k = 0
        difference = 0
        with torch.no_grad():
            for batch in test_loader:
                batch_utterances, utterance_sequence_length, conversation_length_list, padded_labels, padded_speakers, _, _ = batch
                attentive_repre, k_prob = self.ensemble_model(batch)

                for i in range(attentive_repre.shape[0]):
                    dialogue_embedding = attentive_repre[i, :conversation_length_list[i], :]
                    if self.args.train_mode == 'supervised':
                        k_val =  (torch.argmax(k_prob[i, :conversation_length_list[i]]) + 1).item()
                        # k_val = (torch.max(padded_labels[i]) + 1).item()
                    elif self.args.train_mode == 'unsupervised':
                        k_val, _ = utils.calculateK(dialogue_embedding, conversation_length_list[i], self.args.Kmeans_metric, self.device)
                    gold_k = (torch.max(padded_labels[i]) + 1).item()
                    if gold_k == k_val:
                        correct_k = correct_k + 1
                    difference = difference + np.abs(gold_k - k_val)
                    count_k = count_k + 1
                    # print("cluster_number", k_val)
                    cluster_label = KMeans(n_clusters=k_val, random_state=0).fit(dialogue_embedding.cpu().detach().numpy()).labels_
                    cluster_label = utils.order_cluster_labels(cluster_label.tolist())
                    # print("cluster_label after ordering", cluster_label)
                    predicted_labels.append(cluster_label)

                for j in range(len(conversation_length_list)):
                    truth_labels.append(padded_labels[j][:conversation_length_list[j]].tolist())
                
        assert len(predicted_labels) == len(truth_labels)
        
        for (p, t) in zip(predicted_labels, truth_labels):
            # print(p, len(p))
            # print(t, len(t))
            assert len(p) == len(t)

        utils.save_predicted_results(predicted_labels, truth_labels, self.current_time, step_cnt)

        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
        ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')

        accuracy_k = correct_k / count_k
        difference_k = difference / count_k

        return purity_score, nmi_score, ari_score, shen_f_score, accuracy_k, difference_k


    def test(self, test_loader, model_path, step_cnt):
        print("Loading model...")
        self.ensemble_model.load_state_dict(torch.load(model_path))

        predicted_labels = []
        truth_labels = []
        count_k = 0
        correct_k = 0
        difference = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch_utterances, utterance_sequence_length, conversation_length_list, padded_labels, padded_speakers, _, _ = batch
                attentive_repre, k_prob = self.ensemble_model(batch)

                for i in range(attentive_repre.shape[0]):
                    dialogue_embedding = attentive_repre[i, :conversation_length_list[i], :]
                    if self.args.train_mode == 'supervised':
                        k_val =  (torch.argmax(k_prob[i, :conversation_length_list[i]]) + 1).item()
                        # k_val = (torch.max(padded_labels[i]) + 1).item()
                    elif self.args.train_mode == 'unsupervised':
                        k_val, _ = utils.calculateK(dialogue_embedding, conversation_length_list[i], self.args.Kmeans_metric, self.device)
                    gold_k = (torch.max(padded_labels[i]) + 1).item()
                    if gold_k == k_val:
                        correct_k = correct_k + 1
                    difference = difference + np.abs(gold_k - k_val)
                    count_k = count_k + 1
                    cluster_label = KMeans(n_clusters=k_val, random_state=0).fit(dialogue_embedding.cpu().detach().numpy()).labels_
                    cluster_label = utils.order_cluster_labels(cluster_label.tolist())
                    predicted_labels.append(cluster_label)
                
                for j in range(len(conversation_length_list)):
                    truth_labels.append(padded_labels[j][:conversation_length_list[j]].cpu().tolist())
        assert len(predicted_labels) == len(truth_labels)

        utils.save_predicted_results(predicted_labels, truth_labels, self.current_time, step_cnt, mode='test')

        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
        ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')
        accuracy_k = correct_k / count_k
        difference_k = difference / count_k

        log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}, accuracy_k: {}, difference_k: {}".format(
                        round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4), round(accuracy_k, 4), round(difference_k, 4))
        print(log_msg)
