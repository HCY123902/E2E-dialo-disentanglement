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
    speakers = []
    mentions = []

    for item in tqdm(content):
        utterance_list = []
        label_list = []
        speaker_list = []
        speaker_map = {}
        mention_list = []

        for one_uttr in item:
            uttr_content = one_uttr['utterance']
            uttr_word_list = word_tokenize(uttr_content.lower())
            if len(uttr_word_list) > constant.utterance_max_length:
                uttr_word_list = uttr_word_list[:constant.utterance_max_length]
            label = one_uttr['label']

            speaker = one_uttr['speaker']
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)
            speaker = speaker_map.get(speaker)

            mention = np.zeros(constant.dialogue_max_length, dtype=int)
            for s in speaker_map:
                if s in uttr_content:
                    mention[speaker_map[s]] = 1 
            mention[speaker] = 1

            if train_mode == 'unsupervised':
                label = speaker

            speaker_list.append(speaker)
            label_list.append(label)
            utterance_list.append(uttr_word_list)
            mention_list.append(mention.tolist())

            if mode == "train":
                all_utterances.append(copy.deepcopy(utterance_list))
                labels.append(copy.deepcopy(label_list))
                speakers.append(copy.deepcopy(speaker_list))
                mentions.append(copy.deepcopy(mention_list))
        if mode != "train":
            all_utterances.append(copy.deepcopy(utterance_list))
            labels.append(copy.deepcopy(label_list))
            speakers.append(copy.deepcopy(speaker_list))
            mentions.append(copy.deepcopy(mention_list))

    zipped_list = [(a, l, s, m) for (a, l, s, m) in zip(all_utterances, labels, speakers, mentions)]

    random.shuffle(zipped_list)

    all_utterances = [t[0] for t in zipped_list]
    labels = [t[1] for t in zipped_list]
    speakers = [t[2] for t in zipped_list]
    mentions = [t[3] for t in zipped_list]
    
    return all_utterances, labels, speakers, mentions

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

    all_utterances, labels, speakers, mentions = extract_input_data(content, mode, train_mode)
    word_dict = build_word_dict(all_utterances)
    
    return all_utterances, labels, word_dict, speakers, mentions

def read_data(load_var=False, input_=None, mode='train', train_mode='supervised'):
    if load_var:
        all_utterances = utils.save_or_read_input(os.path.join(constant.save_input_path, "{}_utterances.pk".format(mode)))
        labels = utils.save_or_read_input(os.path.join(constant.save_input_path, "{}_labels.pk".format(mode)))
        word_dict = utils.save_or_read_input(os.path.join(constant.save_input_path, "word_dict.pk"))
    else:
        if mode == 'train':
            all_utterances, labels, word_dict, speakers, mentions = read_raw_data(input_, mode, train_mode)
        else:
            all_utterances, labels, _, speakers, mentions = read_raw_data(input_, mode, train_mode)
            if mode == 'dev':
                word_dict = None
            else:
                word_dict = utils.save_or_read_input(os.path.join(constant.save_input_path, "word_dict.pk"))
    return all_utterances, labels, word_dict, speakers, mentions

