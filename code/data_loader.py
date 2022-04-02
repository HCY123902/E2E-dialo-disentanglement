import os
import sys
import numpy as np
from numpy.lib import utils 
import torch 
import random 

from utils import build_batch
from utils import convert_utterances

import constant


class TrainDataLoader(object):
    def __init__(self, all_utterances, labels, word_dict, name='train', add_noise=False, batch_size=constant.batch_size):
        self.all_utterances_batch = [all_utterances[i:i+batch_size] \
                                    for i in range(0, len(all_utterances), batch_size)]
        self.labels_batch = [labels[i:i+batch_size] \
                            for i in range(0, len(labels), batch_size)]
        self.word_dict = word_dict
        self.add_noise = add_noise
        assert len(self.all_utterances_batch) == len(self.labels_batch)
        self.batch_num = len(self.all_utterances_batch)
        print("{} batches created in {} set.".format(self.batch_num, name))

    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= self.batch_num:
            raise IndexError

        utterances = self.all_utterances_batch[key]
        labels = self.labels_batch[key]
        # new_utterance_num_numpy, label_for_loss, new_labels, new_utterance_sequence_length, session_transpose_matrix, \
        #         state_transition_matrix, session_sequence_length, max_conversation_length, loss_mask \
        #                     = build_batch(utterances, labels, self.word_dict, add_noise=self.add_noise)
        # if self.add_noise:
        #     _, label_for_loss, _, _, _, _, _, _, _ = build_batch(utterances, labels, self.word_dict)
        # batch_size, max_length_1, max_length_2 = new_utterance_num_numpy.shape
        # new_utterance_num_numpy = self.convert_to_tensors_1(new_utterance_num_numpy, batch_size, \
        #                                                     max_length_1, max_length_2)
        # batch_size, max_length_1 = loss_mask.shape
        # loss_mask = self.convert_to_tensors_2(loss_mask, batch_size, max_length_1)
        # batch_size, max_length_1 = new_utterance_sequence_length.shape
        # new_utterance_sequence_length = self.convert_to_tensors_2(new_utterance_sequence_length, batch_size, max_length_1)

        # Added
        batch_size = len(utterances)

        conversation_lengths = [len(dialogue) for dialogue in utterances]
        padded_labels = self.convert_to_tensors_label(labels, batch_size, conversation_lengths)

        new_utterance_num_numpy, utterance_sequence_length = self.convert_to_tensors_utterances(utterances, batch_size, conversation_lengths, self.word_dict)

        return new_utterance_num_numpy, utterance_sequence_length, conversation_lengths, padded_labels

    def convert_to_tensors_1(self, utterances, batch_size, max_length, h_size):
        # batch_size, max_conversation_length, max_utterance_length
        if not torch.cuda.is_available():
            new_batch = torch.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
        else:
            new_batch = torch.cuda.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
        # print(new_batch.shape, len(utterances))    
        
        for i in range(len(utterances)):
            for j in range(len(utterances[i])):
                # print(len(utterances[i]))
                # print(len(utterances[i][j]))
                new_batch[i, j][:len(utterances[i][j])] = torch.LongTensor(utterances[i][j])
        return new_batch

    def convert_to_tensors_2(self, batch, batch_size, max_length):
        if not torch.cuda.is_available():
            new_batch = torch.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        else:
            new_batch = torch.cuda.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        for i in range(len(batch)):
            new_batch[i][:len(batch[i])] = torch.LongTensor(batch[i])
        return new_batch

    def convert_to_tensors_utterances(self, batch, batch_size, conversation_lengths, word_dict):
        max_conversation_length = max(conversation_lengths)
        max_utterance_length = max([len(x) for one_uttr in batch for x in one_uttr])

        utterances_num, utterance_sequence_length = convert_utterances(batch, word_dict)

        new_batch = self.convert_to_tensors_1(utterances_num, batch_size, max_conversation_length, max_utterance_length)

        utterance_sequence_length = self.convert_to_tensors_2(utterance_sequence_length, batch_size, max_conversation_length)

        return new_batch, utterance_sequence_length

    def convert_to_tensors_label(self, batch, batch_size, conversation_lengths):
        max_length = max(conversation_lengths)
        if not torch.cuda.is_available():
            new_batch = torch.LongTensor(batch_size, max_length).fill_(-1)
        else:
            new_batch = torch.cuda.LongTensor(batch_size, max_length).fill_(-1)
        for i in range(len(batch)):
            new_batch[i, :conversation_lengths[i]] = torch.LongTensor(batch[i])
        return new_batch
