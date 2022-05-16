import numpy as np 
import torch
import random
import os
import sys
import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm 

import utils
import constant

import copy

def extract_input_data(content, mode, train_mode='supervised'):
    print("Tokenizing sentence and word...")
    all_utterances = []
    labels = []

    # Added for speaker
    speakers = []
    for item in tqdm(content):
        utterance_list = []
        label_list = []

        # Added for speaker
        speaker_list = []
        speaker_map = {}

        for one_uttr in item:
            uttr_content = one_uttr['utterance']
            uttr_word_list = word_tokenize(uttr_content.lower())
            if len(uttr_word_list) > constant.utterance_max_length:
                uttr_word_list = uttr_word_list[:constant.utterance_max_length]
            label = one_uttr['label']

            # Added for speaker
            speaker = one_uttr['speaker']
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)
            speaker = speaker_map.get(speaker)

            # Added for unsupervised training
            if train_mode == 'unsupervised':
                label = speaker

            speaker_list.append(speaker)
            label_list.append(label)
            utterance_list.append(uttr_word_list)

            # Added
            if mode == "train":
                all_utterances.append(copy.deepcopy(utterance_list))
                labels.append(copy.deepcopy(label_list))
                speakers.append(copy.deepcopy(speaker_list))
            
#             if mode != "train" and (len(utterance_list) == 1 or len(utterance_list) == len(item) // 2 or (len(utterance_list) >= len(item) * 0.94 and len(utterance_list) < len(item))):
#                 all_utterances.append(copy.deepcopy(utterance_list))
#                 labels.append(copy.deepcopy(label_list))
#             if mode != "train" and (len(utterance_list) <= 1 or len(utterance_list) == len(item) // 2 or (len(utterance_list) >= len(item) * 0.94 and len(utterance_list) < len(item))):
            if mode != "train":
                all_utterances.append(copy.deepcopy(utterance_list))
                labels.append(copy.deepcopy(label_list))
                speakers.append(copy.deepcopy(speaker_list))

        # all_utterances.append(utterance_list)
        # labels.append(label_list)
    
#     if mode == "train":
    zipped_list = [(a, l, s) for (a, l, s) in zip(all_utterances, labels, speakers)]

    random.shuffle(zipped_list)

    all_utterances = [t[0] for t in zipped_list]
    labels = [t[1] for t in zipped_list]
    speakers = [t[2] for t in zipped_list]

    # Added
    all_utterances = all_utterances[:len(all_utterances)//10]
    labels = labels[:len(all_utterances)]
    speakers = speakers[:len(all_utterances)]
    
    return all_utterances, labels, speakers

def build_word_dict(all_utterances):
    print("Building word dictionary...")
    word_dict = dict()
    word_dict['<PAD>'] = constant.PAD_ID
    word_dict['<UNK>'] = constant.UNK_ID

    word_cnt = dict()
    for one_case in all_utterances:
        for one_uttr in one_case:
            for word in one_uttr:
                if word not in word_cnt:
                    word_cnt[word] = 0
                word_cnt[word] += 1
    for key, val in word_cnt.items():
       if val > 10:
            word_dict[key] = len(word_dict)

    print("{} words in total and {} words in the dictionary".format(len(word_cnt), len(word_dict)))
    return word_dict

def read_raw_data(datapath, mode='train', train_mode='supervised'):
    print("Reading {} data...".format(mode))
    with open(datapath) as fin:
        content = json.load(fin)
    print("{} {} data examples read.".format(len(content), mode))

    all_utterances, labels, speakers = extract_input_data(content, mode, train_mode)
    word_dict = build_word_dict(all_utterances)
    
    return all_utterances, labels, word_dict, speakers

def read_data(load_var=False, input_=None, mode='train', train_mode='supervised'):
    if load_var:
        all_utterances = utils.save_or_read_input(os.path.join(constant.save_input_path, "{}_utterances.pk".format(mode)))
        labels = utils.save_or_read_input(os.path.join(constant.save_input_path, "{}_labels.pk".format(mode)))
        word_dict = utils.save_or_read_input(os.path.join(constant.save_input_path, "word_dict.pk"))
    else:
        if mode == 'train':
            all_utterances, labels, word_dict, speakers = read_raw_data(input_, mode, train_mode)
        else:
            all_utterances, labels, _, speakers = read_raw_data(input_, mode, train_mode)
            if mode == 'dev':
                word_dict = None
            else:
                word_dict = utils.save_or_read_input(os.path.join(constant.save_input_path, "word_dict.pk"))
    return all_utterances, labels, word_dict, speakers

