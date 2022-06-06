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
    def __init__(self, all_utterances, labels, word_dict, speakers, mentions, name='train', add_noise=False, batch_size=constant.batch_size, train_mode = 'supervised'):
        self.all_utterances_batch = [all_utterances[i:i+batch_size] \
                                    for i in range(0, len(all_utterances), batch_size)]
        self.labels_batch = [labels[i:i+batch_size] \
                            for i in range(0, len(labels), batch_size)]
        # Added for speaker
        self.speakers_batch = [speakers[i:i+batch_size] \
                            for i in range(0, len(labels), batch_size)]
        self.mentions_batch = [mentions[i:i+batch_size] \
                            for i in range(0, len(labels), batch_size)]
        self.word_dict = word_dict
        self.add_noise = add_noise
        assert len(self.all_utterances_batch) == len(self.labels_batch)
        self.batch_num = len(self.all_utterances_batch)
        print("{} batches created in {} set.".format(self.batch_num, name))
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.train_mode = train_mode
        # if self.train_mode == 'unsupervised':
        # Postivie and negative sampling for triplet
        print("Start generating positive and negative samples")
        self.pos_samples = []
        self.neg_samples = []
        for key, batch in enumerate(self.labels_batch):
            batch_pos_samples = []
            batch_neg_samples = []
            max_conversation_length = max([len(dialogue) for dialogue in self.all_utterances_batch[key]])
            for (i, dialogue_labels) in enumerate(batch):
                dialgoue_labels = np.array(dialogue_labels)
                
                if self.train_mode == 'unsupervised':
                    neg_pool = np.array([[k, p] for k in range(len(batch)) for p in range(len(batch[k])) if k != i])
                    
                dialogue_pos_samples = []
                dialogue_neg_samples = []
                # Sample for every utterance label
                for label in dialgoue_labels:
                    start = i * max_conversation_length
                    pos_pool = np.where(dialgoue_labels == label)[0]
                    # scalar
                    if self.train_mode == 'unsupervised':
                        pos_sample = pos_pool + start
                        
                        neg_sample = neg_pool[np.random.choice(len(neg_pool), min(100, len(neg_pool)), replace=False)]
                        # [conversation_length, num_samples]
                        neg_sample = neg_sample[:, 0] * max_conversation_length + neg_sample[:, 1]
                    else:
                        pos_sample = pos_pool + start
                        neg_sample = np.where(dialgoue_labels != label)[0] + start

                    dialogue_pos_samples.append(pos_sample)
                    dialogue_neg_samples.append(neg_sample)
                    
                # if self.train_mode == 'unsupervised':
                #     dialogue_pos_samples = np.array(dialogue_pos_samples)
                #     dialogue_neg_samples = np.array(dialogue_neg_samples)
                # print(dialogue_neg_samples.shape)
                batch_pos_samples.append(dialogue_pos_samples)
                batch_neg_samples.append(dialogue_neg_samples)
            self.pos_samples.append(batch_pos_samples)
            self.neg_samples.append(batch_neg_samples)
        print("Sample generation completed")

    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= self.batch_num:
            raise IndexError

        utterances = self.all_utterances_batch[key]
        labels = self.labels_batch[key]

        # Added for speaker
        # speakers = self.speakers_batch[key]
        mentions = self.mentions_batch[key]

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

        # Added for speaker
        # padded_speakers = self.convert_to_tensors_speaker(speakers, batch_size, conversation_lengths)
        # padded_speakers = padded_speakers.to(self.device, dtype=torch.float)
        padded_speakers = self.convert_to_tensors_mention(mentions, batch_size, conversation_lengths)
        padded_speakers = padded_speakers.to(self.device, dtype=torch.float)

        new_utterance_num_numpy, utterance_sequence_length = self.convert_to_tensors_utterances(utterances, batch_size, conversation_lengths, self.word_dict)

        pos_mask = None
        sample_mask = None
        # if self.train_mode == 'unsupervised':
        # Added for sampling
        pos_sample = self.pos_samples[key]
        neg_sample = self.neg_samples[key]
        pos_mask, sample_mask = self.create_mask(pos_sample, neg_sample, batch_size, max(conversation_lengths))

        return new_utterance_num_numpy, utterance_sequence_length, conversation_lengths, padded_labels, padded_speakers, pos_mask, sample_mask

    def convert_to_tensors_1(self, utterances, batch_size, max_length, h_size):
        # batch_size, max_conversation_length, max_utterance_length
        # if not torch.cuda.is_available():
        #     new_batch = torch.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
        #     for i in range(len(utterances)):
        #         for j in range(len(utterances[i])):
        #             # print(len(utterances[i]))
        #             # print(len(utterances[i][j]))
        #             new_batch[i, j][:len(utterances[i][j])] = torch.LongTensor(utterances[i][j])
        # else:
        new_batch = torch.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
    # print(new_batch.shape, len(utterances))    
    
        for i in range(len(utterances)):
            for j in range(len(utterances[i])):
                # print(len(utterances[i]))
                # print(len(utterances[i][j]))
                new_batch[i, j][:len(utterances[i][j])] = torch.LongTensor(utterances[i][j])
        return new_batch.to(self.device)

    def convert_to_tensors_2(self, batch, batch_size, max_length):
        # if not torch.cuda.is_available():
        #     new_batch = torch.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        #     for i in range(len(batch)):
        #         new_batch[i][:len(batch[i])] = torch.LongTensor(batch[i])
        # else:
        new_batch = torch.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        for i in range(len(batch)):
            new_batch[i][:len(batch[i])] = torch.LongTensor(batch[i])
        return new_batch.to(self.device)

    def convert_to_tensors_utterances(self, batch, batch_size, conversation_lengths, word_dict):
        # TODO: investigate whethere it is possible to replace batch level max length with constant.max_length when intializing the tensors
        max_conversation_length = max(conversation_lengths)
        max_utterance_length = max([len(x) for one_uttr in batch for x in one_uttr])

        utterances_num, utterance_sequence_length = convert_utterances(batch, word_dict)

        new_batch = self.convert_to_tensors_1(utterances_num, batch_size, max_conversation_length, max_utterance_length)

        utterance_sequence_length = self.convert_to_tensors_2(utterance_sequence_length, batch_size, max_conversation_length)

        return new_batch, utterance_sequence_length

    def convert_to_tensors_label(self, batch, batch_size, conversation_lengths):
        max_length = max(conversation_lengths)
        # if not torch.cuda.is_available():
        #     new_batch = torch.LongTensor(batch_size, max_length).fill_(-1)
        #     for i in range(len(batch)):
        #         new_batch[i, :conversation_lengths[i]] = torch.LongTensor(batch[i])
        # else:
        new_batch = torch.LongTensor(batch_size, max_length).fill_(-1)
        for i in range(len(batch)):
            new_batch[i, :conversation_lengths[i]] = torch.LongTensor(batch[i])

        return new_batch.to(self.device)

    def convert_to_tensors_speaker(self, batch, batch_size, conversation_lengths):
        max_length = max(conversation_lengths)
        # if not torch.cuda.is_available():
        #     new_batch = torch.LongTensor(batch_size, max_length).fill_(-1)
        #     for i in range(len(batch)):
        #         new_batch[i, :conversation_lengths[i]] = torch.LongTensor(batch[i])
        # else:
        new_batch = torch.zeros(batch_size, max_length, constant.dialogue_max_length).float()
        for i in range(len(batch)):
            for j in range(conversation_lengths[i]):
                new_batch[i, j, batch[i][j]] = 1.0

        return new_batch.to(self.device)
    
    def convert_to_tensors_mention(self, batch, batch_size, conversation_lengths):
        max_length = max(conversation_lengths)
        # if not torch.cuda.is_available():
        #     new_batch = torch.LongTensor(batch_size, max_length).fill_(-1)
        #     for i in range(len(batch)):
        #         new_batch[i, :conversation_lengths[i]] = torch.LongTensor(batch[i])
        # else:
        new_batch = torch.zeros(batch_size, max_length, constant.dialogue_max_length).float()
        for i in range(len(batch)):
            new_batch[i, :conversation_lengths[i]] = torch.tensor(batch[i]).float()

        return new_batch.to(self.device)

    def create_mask(self, pos_sample, neg_sample, batch_size, max_conversation_length):
        pos_masks = torch.zeros(batch_size, max_conversation_length, batch_size * max_conversation_length).float().to(self.device)
        sample_masks = torch.zeros(batch_size, max_conversation_length, batch_size * max_conversation_length).float().to(self.device)

        for i in range(batch_size):
            # In criterion, batch embeddings will be stretched to shape [batch_size * max_conversation_length, hidden_size] 
            # start = i * max_conversation_length
            # pos_sample: [batch_size, conversation_length] -> pos_sample[i]: conversation_length -> [conversation_length, 1]
            # dialogue_pos_samples = (torch.LongTensor(pos_sample[i]).to(self.device).unsqueeze(0)).T
            
            # if self.train_mode == 'unsupervised':
            #     dialogue_pos_samples = (torch.LongTensor(pos_sample[i]).to(self.device))
            #     pos_masks[i].scatter_(dim=1, index=dialogue_pos_samples, value=1.0)

            #     sample_masks[i].scatter_(dim=1, index=dialogue_pos_samples, value=1.0)

                # neg_samples: [batch_size, conversation_length, num_samples, 2] -> [conversation_length, num_samples]
                # print("neg_sample", len(neg_sample[i]), len(neg_sample[i][0]))
                # for utterance_samples in neg_sample[i]:
                #     print("a", utterance_samples[:, 0] * max_conversation_length)
                #     print("b", utterance_samples[:, 1])
                #     print("combined", (utterance_samples[:, 0] * max_conversation_length + utterance_samples[:, 1]).shape)
                # neg_sample_position = np.array([(utterance_samples[:, 0] * max_conversation_length + utterance_samples[:, 1]).reshape(-1) for utterance_samples in neg_sample[i]])
                # neg_sample_position = neg_sample[i][:, :, 0] *  + neg_sample[i][:, :, 1]

                # print("max_conversation_length", max_conversation_length)
                # print("neg_sample_position", neg_sample_position.shape)
                # print("neg_sample", neg_sample[i].shape)
            #     dialogue_neg_samples = torch.LongTensor(neg_sample[i]).to(self.device)
            #     sample_masks[i].scatter_(dim=1, index=dialogue_neg_samples, value=1.0)
            # else:
                # Iterate for each utterance
                for j, s in enumerate(pos_sample[i]):
                    pos_masks[i, j, s] = 1.0
                    sample_masks[i, j, s] = 1.0
                for j, s in enumerate(neg_sample[i]):
                    sample_masks[i, j, s] = 1.0

        return pos_masks.to(self.device, dtype=torch.float), sample_masks.to(self.device, dtype=torch.float)

