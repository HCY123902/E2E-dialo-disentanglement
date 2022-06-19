import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

import random
import numpy as np 

import constant


class EnsembleModel(nn.Module):
    def __init__(self, word_dict, word_emb=None, bidirectional=False):
        super(EnsembleModel, self).__init__()
        self.utterance_encoder = UtteranceEncoder(word_dict, word_emb=word_emb, bidirectional=bidirectional, \
                                            n_layers=1, input_dropout=0, dropout=0, rnn_cell='lstm')
        self.attentive_encoder = SelfAttentiveEncoder()
        self.conversation_encoder = ConversationEncoder(bidirectional=bidirectional, n_layers=1, dropout=0, rnn_cell='lstm')
        self.conversation_attentive_encoder = ConversationAttentiveEncoder()
        self.k_predictor = torch.nn.Sequential(
            torch.nn.Linear(constant.attention_size + 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, constant.state_num)
        )

        self.m = torch.nn.Softmax(dim=1)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    
    def forward(self, batch):
        batch_utterances, utterance_sequence_length, conversation_lengths, padded_labels, padded_speakers, _, _ = batch
        utterance_repre, shape = self.utterance_encoder(batch_utterances, utterance_sequence_length)
        # [batch_size, max_conversation_length, hidden_size]
        attentive_repre = self.attentive_encoder(batch_utterances, utterance_repre, shape)

        # [batch_size, max_conversation_length, hidden_size]
        conversation_repre = self.conversation_encoder(attentive_repre)
        batch_size = attentive_repre.size(0)

        if constant.adopt_speaker:
            conversation_repre = torch.cat((conversation_repre, padded_speakers), dim=-1)
    
        # k_logtis = self.k_predictor(torch.cat((self.pad_dialogue(attentive_repre).reshape(batch_size, -1), self.pad_speaker(padded_speakers)), dim=1))
        conversation_attention = self.conversation_attentive_encoder(batch_utterances, conversation_repre, shape)
        num_speakers = torch.sum((torch.sum(padded_speakers, dim=1) != 0), dim=1, keepdim=True).to(self.device, dtype=torch.float)
        conversation_len = torch.tensor(conversation_lengths).unsqueeze(1).to(self.device, dtype=torch.float)
        k_logits = self.k_predictor(torch.cat((conversation_attention, num_speakers, conversation_len), dim=1))
        
        k_prob = self.m(k_logits)

        return conversation_repre, k_prob
        # return attentive_repre, k_prob

    def pad_dialogue(self, attentive_repre):
        s = attentive_repre.size()
        padded = F.pad(attentive_repre, (0, constant.utterance_max_length - s[2], 0, constant.dialogue_max_length - s[1]), "constant", 0)
        return padded

    def pad_speaker(self, speakers):
        s = speakers.size()
        padded = F.pad(speakers, (0, constant.dialogue_max_length - s[1]), "constant", 0)
        return padded

class UtteranceEncoder(nn.Module):
    def __init__(self, word_dict, word_emb=None, bidirectional=False, n_layers=1, input_dropout=0, \
                        dropout=0, rnn_cell='lstm'):
        super(UtteranceEncoder, self).__init__()
        self.word_emb = word_emb
        self.word_emb_matrix = nn.Embedding(len(word_dict), constant.embedding_size)
        self.init_embedding()
        # self.bidirectional = bidirectional
        # self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.bidirectional = bidirectional
        
        # bi = 2 if self.bidirectional else 1
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(constant.embedding_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(constant.embedding_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        if self.bidirectional:
            self.bidirectional_projection = nn.Sequential(
                nn.Linear(constant.hidden_size*2, constant.hidden_size), 
                nn.ReLU()
            )

    def init_embedding(self):
        if self.word_emb is None:
            self.word_emb_matrix.weight.data.uniform_(-0.1, 0.1)
        else:
            self.word_emb_matrix.weight.data.copy_(torch.from_numpy(self.word_emb))
    
    def forward(self, input_var, input_lens):
        shape = input_var.size() # batch_size, max_conversation_length, max_utterance_length
        input_var = input_var.view(-1, shape[2])
        input_lens = input_lens.reshape(-1)
        embeded_input = self.word_emb_matrix(input_var)
        word_output, _ = self.encoder(embeded_input)
        # word_output: [batch_size * max_conversation_length, max_utterance_length, hidden_size]
        if self.bidirectional:
            word_output = self.bidirectional_projection(word_output)
        return word_output, shape

# Attention is on the utterance level
class SelfAttentiveEncoder(nn.Module):
    def __init__(self, dropout=0.):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(constant.hidden_size, constant.hidden_size, bias=False)
        self.ws2 = nn.Linear(constant.hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = 1

    def forward(self, inp, lstm_output, shape):
        size = lstm_output.size()  # [batch_size * max_conversation, max_utterance_length, hidden_size]
        compressed_embeddings = lstm_output.contiguous().view(-1, size[2])  # [batch_size * max_conversation_length * max_utterance_length, hidden_size]
        transformed_inp = inp.view(size[0], 1, size[1])  # [batch_size * max_conversation_length, 1, max_utterance_length]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [batch_size * max_conversation_length, hop, max_utterance_length]

        hbar = self.tanh(self.ws1(compressed_embeddings)) # [batch_size * max_conversation_length * max_utterance_length, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [batch_size * max_conversation_length, max_utterance_length, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [batch_size * max_conversation_length, hop, max_utterance_length]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == constant.PAD_ID).float())
            # [batch_size * max_conversation_length, hop, max_utterance_length] + [batch_size * max_conversation_length, hop, max_utterance_length]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [batch_size * max_conversation_length * hop, max_utterance_length]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [batch_size * max_conversation_length, hop, max_utterance_length]
        # [batch_size * max_conversation_length, hop, hidden_size]
        ret_output = torch.bmm(alphas, lstm_output).squeeze().view(shape[0], shape[1], constant.hidden_size) 
        return ret_output


class ConversationEncoder(nn.Module):
    def __init__(self, bidirectional=False, n_layers=1, dropout=0, rnn_cell='lstm'):
        super(ConversationEncoder, self).__init__()
        self.bidirectional = bidirectional
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        if self.bidirectional:
            self.bidirectional_projection = nn.Sequential(
                nn.Linear(constant.hidden_size*2, constant.hidden_size), 
                nn.ReLU()
            )
    
    def forward(self, input_var):
        # input_var: [batch_size, max_conversation_length, hidden_size]
        conv_output, _ = self.encoder(input_var)
        # conv_output: [batch_size, max_conversation_length, hidden_size]
        if self.bidirectional:
            conv_output = self.bidirectional_projection(conv_output)
        return conv_output


# Attention is on the conversation level
class ConversationAttentiveEncoder(nn.Module):
    def __init__(self, dropout=0.):
        super(ConversationAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(constant.attention_size, constant.attention_size, bias=False)
        self.ws2 = nn.Linear(constant.attention_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp, lstm_output, shape):
        alphas = self.ws2(self.tanh(self.ws1(lstm_output))).squeeze(2) # [batch_size, max_conversation_length, 1] -> [batch_size, max_conversation_length]
        transformed_inp = inp[:, :, 0] # [batch_size, max_conversation_length]

        penalized_alphas = alphas + (
            -10000 * (transformed_inp == constant.PAD_ID).float())

        alphas = (self.softmax(penalized_alphas)).unsqueeze(1)  # [batch_size, 1, max_conversation_length]

        # [batch_size, 1, max_conversation_length] * [batch_size, max_conversation_length, hidden_size] -> [batch_size, hidden_size]
        ret_output = torch.bmm(alphas, lstm_output).squeeze()
        return ret_output

class SessionEncoder(nn.Module):
    def __init__(self, bidirectional=False, n_layers=1, dropout=0, rnn_cell='lstm'):
        super(SessionEncoder, self).__init__()
        self.bidirectional = bidirectional
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(constant.hidden_size, constant.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        if self.bidirectional:
            self.bidirectional_projection = nn.Sequential(
                nn.Linear(constant.hidden_size*2, constant.hidden_size), 
                nn.ReLU()
            )
    
    def forward(self, input_var, transpose_matrix):
        # input_var: [batch_size, max_conversation_length, hidden_size]
        batch_size, max_conversation_length, _ = input_var.size()
        input_var = input_var.contiguous().view(-1, constant.hidden_size)
        input_var = input_var[transpose_matrix]
        # [batch_size * max_conversation_length, hidden_size]
        input_var = input_var.view(-1, int(max_conversation_length/(constant.state_num-1)), constant.hidden_size)
        # [batch_size * 4, max_session_length, hidden_size]
        output, _ = self.encoder(input_var)
        if self.bidirectional:
            output = self.bidirectional_projection(output)
        # output: [batch_size*4, max_session_length, hidden_size]
        output = output.view(batch_size, constant.state_num-1, int(max_conversation_length/(constant.state_num-1)), constant.hidden_size)
        return output


