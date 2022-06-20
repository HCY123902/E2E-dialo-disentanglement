import os
import sys
import argparse
from tqdm import tqdm 
import torch
import random
import logging
import ast
from time import strftime, gmtime
import pickle
import re

import utils
from data_processing import read_data
from utils import build_embedding_matrix
from data_loader import TrainDataLoader
from supervised_trainer import SupervisedTrainer
from models import UtteranceEncoder, ConversationEncoder, SelfAttentiveEncoder, EnsembleModel
import constant


random.seed(constant.seed)
torch.manual_seed(constant.seed)

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)

current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
log_head = "Learning Rate: {}; Random Seed: {}; ".format(constant.learning_rate, constant.seed)


def train(args):
    utils.make_all_dirs(current_time)
    if args.load_var: 
        all_utterances, labels, word_dict, speakers, mentions = read_data(load_var=args.load_var, input_=None, mode='train', train_mode=args.train_mode)
        dev_utterances, dev_labels, _, dev_speakers, dev_mentions = read_data(load_var=args.load_var, input_=None, mode='dev', train_mode=args.train_mode)
    else:
        all_utterances, labels, word_dict, speakers, mentions = read_data(load_var=args.load_var, \
                input_=os.path.join(constant.data_path, "entangled_train.json"), mode='train', train_mode=args.train_mode)
        dev_utterances, dev_labels, _, dev_speakers, dev_mentions = read_data(load_var=args.load_var, \
                input_=os.path.join(constant.data_path, "entangled_dev.json"), mode='dev', train_mode=args.train_mode)
            
    word_emb = build_embedding_matrix(word_dict, glove_loc=args.glove_loc, \
                    emb_loc=os.path.join(constant.save_input_path, "word_emb.pk"), load_emb=False)
    
    if args.save_input:
        utils.save_or_read_input(os.path.join(constant.save_input_path, "train_utterances.pk"), \
                                    rw='w', input_obj=all_utterances)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "train_labels.pk"), \
                                    rw='w', input_obj=labels)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "word_dict.pk"), \
                                    rw='w', input_obj=word_dict)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "word_emb.pk"), \
                                    rw='w', input_obj=word_emb)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "dev_utterances.pk"), \
                                    rw='w', input_obj=dev_utterances)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "dev_labels.pk"), \
                                    rw='w', input_obj=dev_labels)
     
    train_dataloader = TrainDataLoader(all_utterances, labels, word_dict, speakers, mentions, train_mode = args.train_mode)

    dev_dataloader = TrainDataLoader(dev_utterances, dev_labels, word_dict, dev_speakers, dev_mentions, name='dev', train_mode = args.train_mode)
    
    logger_name = os.path.join(constant.log_path, "{}.txt".format(current_time))
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()

    ensemble_model = EnsembleModel(word_dict, word_emb=word_emb, bidirectional=True)

    if torch.cuda.is_available():
        ensemble_model.cuda()

    supervised_trainer = SupervisedTrainer(args, ensemble_model, \
                                                logger=logger, current_time=current_time)
    
    supervised_trainer.train(train_dataloader, dev_dataloader)


def test(args):
    if args.load_var:
        test_utterances, test_labels, word_dict, test_speakers, test_mentions = read_data(load_var=args.load_var, input_=None, mode='test')
    else:
        test_utterances, test_labels, word_dict, test_speakers, test_mentions = read_data(load_var=args.load_var, \
                input_=os.path.join(constant.data_path, "entangled_{}.json".format(args.mode)), mode='test')
    
    word_emb = build_embedding_matrix(word_dict, glove_loc=args.glove_loc, \
                    emb_loc=os.path.join(constant.save_input_path, "word_emb.pk"), load_emb=False)
    
    if args.save_input:
        utils.save_or_read_input(os.path.join(constant.save_input_path, "{}_utterances.pk".format(args.mode)), \
                                    rw='w', input_obj=test_utterances)
        utils.save_or_read_input(os.path.join(constant.save_input_path, "{}_labels.pk".format(args.mode)), \
                                    rw='w', input_obj=test_labels)
    
    current_time = re.findall('.*model_(.+?)/.*', args.model_path)[0]
    step_cnt = re.findall('.step_(.+?)\.pkl', args.model_path)[0]

    test_dataloader = TrainDataLoader(test_utterances, test_labels, word_dict, test_speakers, test_mentions, name='test', batch_size=16, train_mode = args.train_mode)
    
    ensemble_model = EnsembleModel(word_dict, word_emb=word_emb, bidirectional=True)
    if torch.cuda.is_available():
        ensemble_model.cuda()

    supervised_trainer = SupervisedTrainer(args, ensemble_model, current_time=current_time)
    
    supervised_trainer.test(test_dataloader, args.model_path, step_cnt=step_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--save_input", action='store_true')
    parser.add_argument("--load_var", action='store_true')
    parser.add_argument('--glove_loc', type=str, default=constant.glove_path)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--Kmeans_metric', type=str, choices=['silhouette', 'elbow', 'combined'], default='elbow')
    parser.add_argument('--print_detail', action='store_true')
    parser.add_argument('--train_mode', type=str, choices=['supervised', 'unsupervised'], default='supervised')
    parser.add_argument('--adopt_speaker', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test' or args.mode == 'dev':
        test(args)
    else:
        raise ValueError('Mode Error')



    
