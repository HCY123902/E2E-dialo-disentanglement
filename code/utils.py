import torch
import json
import os
import sys
import numpy as np 
import pickle
from collections import Counter
import random
import copy
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import constant

import kmeans_pytorch

import warnings
warnings.filterwarnings("ignore")

def make_all_dirs(current_time):
    if not os.path.exists(constant.log_path):
        os.makedirs(constant.log_path)
    if not os.path.exists(constant.save_input_path):
        os.makedirs(constant.save_input_path)
    if not os.path.exists(os.path.join(constant.output_path, 'result_'+current_time)):
        os.makedirs(os.path.join(constant.output_path, 'result_'+current_time))
    if not os.path.exists(constant.output_path):
        os.makedirs(constant.output_path)
    if not os.path.exists(os.path.join(constant.save_model_path, 'model_'+current_time)):
        os.makedirs(os.path.join(constant.save_model_path, 'model_'+current_time))


def save_predicted_results(predicted_labels, truth_labels, current_time, test_step, mode='dev'):
    filepath = os.path.join(constant.output_path, 'result_'+current_time)
    predicted_filename = os.path.join(filepath, 'predicted_step_{}.pkl'.format(test_step) if mode == 'dev' else 'test_predicted_step_{}.pkl'.format(test_step))
    truth_filename = os.path.join(filepath, 'truth.pkl' if mode == 'dev' else 'truth_test.pkl')
    if not os.path.exists(truth_filename):
        with open(truth_filename, 'wb') as fout:
            pickle.dump(truth_labels, fout)
    if not os.path.exists(predicted_filename):
        with open(predicted_filename, 'wb') as fout:
            pickle.dump(predicted_labels, fout)


def save_or_read_input(path, rw='r', input_obj=None):
    if rw == 'w':
        with open(path, 'wb') as fout:
            pickle.dump(input_obj, fout)
        print("{} saved successfully!".format(path))
    elif rw == 'r':
        with open(path, 'rb') as fin:
            ret_val = pickle.load(fin)
        print("{} read successfully!".format(path))
        return ret_val


def build_embedding_matrix(word_dict, glove_loc=None, emb_loc=None, load_emb=False):
    print("Building word embedding matrix...")
    if load_emb:
        with open(emb_loc, 'rb') as fin:
            word_emb = pickle.load(fin)
    else:
        word_emb = np.random.uniform(-1, 1, (len(word_dict), constant.embedding_size))
        tokens = word_dict.keys()
        line_cnt = 0
        oov_cnt = 0
        with open(glove_loc) as fin:
            for line in fin:
                line_cnt += 1
                if line_cnt % 500000 == 0:
                    percent = line_cnt/2196017
                    percent = "%.2f%%" % (percent*100)
                    print("{} of glove read.".format(percent))
                splitted_line = line.strip().split(' ')
                word = splitted_line[0]
                assert len(splitted_line[-constant.embedding_size:]) == constant.embedding_size
                if word in tokens:
                    word_emb[word_dict[word]] = [float(v) for v in splitted_line[-constant.embedding_size:]]
                else:
                    oov_cnt += 1
        print("{} out of {} words are OOVs.".format(oov_cnt, len(word_dict)))
    print("Building word embeding matrix over.")
    return word_emb

def convert_utterances(utterances, word_dict):
    utterances_num = []
    utterance_sequence_length = [] # Sequence length of the batch
    for i in range(len(utterances)):
        one_instance = []
        one_uttr_sequence_length = []
        for one_uttr in utterances[i]:
            one_uttr_num = []
            one_uttr_sequence_length.append(len(one_uttr))
            for word in one_uttr:
                one_uttr_num.append(word_dict.get(word, constant.UNK_ID))
            one_instance.append(one_uttr_num)
        utterances_num.append(one_instance)
        utterance_sequence_length.append(one_uttr_sequence_length)
    return utterances_num, utterance_sequence_length

def calculate_purity_scores(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


def calculate_shen_f_score(y_true, y_pred):
    def get_f_score(i, j, n_i_j, n_i, n_j):
        recall = n_i_j / n_i
        precision = n_i_j / n_j
        if recall == 0 and precision == 0:
            f_score = 0.
        else:
            f_score = 2 * recall * precision / (recall + precision)
        return f_score
    
    y_true_cnt = dict(Counter(y_true))
    y_pred_cnt = dict(Counter(y_pred))
    y_pred_dict = dict()
    for i, val in enumerate(y_pred):
        if y_pred_dict.get(val, None) == None:
            y_pred_dict[val] = dict()
        if y_pred_dict[val].get(y_true[i], None) == None:
            y_pred_dict[val][y_true[i]] = 0
        y_pred_dict[val][y_true[i]] += 1
    shen_f_score = 0.
    for i, val_i in y_true_cnt.items():
        f_list = []
        for j, val_j in y_pred_cnt.items():
            f_list.append(get_f_score(i, j, y_pred_dict[j].get(i, 0), val_i, val_j))
        shen_f_score += max(f_list) * y_true_cnt[i] / len(y_true)
    return shen_f_score


def compare(predicted_labels, truth_labels, metric):
    if metric == 'purity':
        purity_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            purity_scores.append(calculate_purity_scores(y_true, y_pred))
        return sum(purity_scores)/len(purity_scores)
    elif metric == 'NMI':
        NMI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            NMI_scores.append(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
        return sum(NMI_scores)/len(NMI_scores)
    elif metric == 'ARI':
        ARI_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            ARI_scores.append(metrics.adjusted_rand_score(y_true, y_pred))
        return sum(ARI_scores)/len(ARI_scores)
    elif metric == "shen_f":
        f_scores = []
        for y_true, y_pred in zip(truth_labels, predicted_labels):
            f_scores.append(calculate_shen_f_score(y_true, y_pred))
        return sum(f_scores)/len(f_scores)


def order_cluster_labels(cluster_labels):
    ordered_labels = []
    record = {}
    for label in cluster_labels:
        if label not in record:
            record[label] = len(record)
        ordered_labels.append(record[label])

    return ordered_labels

def calculateK(dialogue_embedding, dialogue_length, method, device):
    average_K = max(int((dialogue_length / float(constant.dialogue_max_length)) * (constant.state_num)), 1)
    if dialogue_length <= 2:
        print("Returning average K as K since there are at most 2 utterances in this dialogue")
        return average_K, np.zeros(dialogue_length, dtype=int)
    n  = 1
    if method == 'silhouette':
        scores = []
        for K in range(2, min(dialogue_length - 1, constant.state_num) + 1):
            cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(X=dialogue_embedding, num_clusters=K, distance='euclidean', device=device, tqdm_flag=False)
            labels = cluster_ids_x.cpu().detach().numpy()
            try:
                scores.append([K, silhouette_score(dialogue_embedding.cpu().detach().numpy(), labels), labels])
            except Exception as e:
                print(e)
                print("Returning average K as K")
                if average_K <= 1:
                    return average_K, np.zeros(dialogue_length, dtype=int)
                cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(X=dialogue_embedding, num_clusters=average_K, distance='euclidean', device=device, tqdm_flag=False)
                return average_K, cluster_ids_x.cpu().detach().numpy()
            
        if n == 1:
            m = max(scores, key=lambda x:x[1])
            return m[0], m[2]

        scores.sort(key=lambda x:x[1], reverse=True)

        # Select the K closer to average_K
        scores = [(i[0], np.abs(i[0] - average_K), i[2]) for i in scores[:n]]
        m = min(scores, key=lambda x:x[1])
        return m[0], m[2]

    elif method == 'elbow':
        scores = []
        closest_centers = torch.mean(dialogue_embedding, dim=0).repeat(dialogue_length, 1).to(device)
        # inertia = torch.linalg.norm(dialogue_embedding.cpu()-closest_centers, dim=1, ord=2)
        inertia = (torch.square(dialogue_embedding-closest_centers)).sum().item()
        scores.append(np.array([1, inertia, np.zeros(dialogue_length, dtype=int)]))
        for K in range(2, min(dialogue_length, constant.state_num) + 1):
            cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(X=dialogue_embedding, num_clusters=K, distance='euclidean', device=device, tqdm_flag=False)
            labels = cluster_ids_x.cpu().detach().numpy()
            closest_centers = cluster_centers[cluster_ids_x].to(device)
            inertia = (torch.square(dialogue_embedding-closest_centers)).sum().item()

            scores.append(np.array([K, inertia, labels]))

        rate = [(scores[i][0], calculate_angle(scores[i-1][:2], scores[i][:2], scores[i+1][:2]), scores[i][2]) for i in range(1, len(scores) - 1)]
        if n == 1:
            m = min(rate, key=lambda x:x[1])
            return int(m[0]), m[2]

        rate.sort(key=lambda x:x[1])

        # Select the K closer to average_K
        rate = [(i[0], np.abs(i[0] - average_K), i[2]) for i in rate[:n]]
        m = min(rate, key=lambda x:x[1])
        return m[0], m[2]

    elif method == 'combined':
        # TODO
        print("Method not implemented, returning the average K")
        return average_K, None

    else:
        print("Method not defined, returning the average K")
        return average_K, None
    
def calculate_angle(p1, p2, p3):
    e12 = np.linalg.norm(p1 - p2)
    e23 = np.linalg.norm(p2 - p3)
    e13 = np.linalg.norm(p1 - p3)
    return np.arccos((e12**2 + e23**2 - e13**2)/(2*e12*e23))

