# coding=utf-8
import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='9'
import random

import numpy as np
import pandas as pd
import torch
from transformers import (BertConfig, BertForTokenClassification,
                          BertTokenizer)
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs
from model import (Aspect_Text_GAT_ours,
                   Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only)
from ner_utils import NerUtils
from trainer import train

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_name', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Choose absa dataset.')
    parser.add_argument('--output_dir', type=str, default='/data/output-gcn',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')

    parser.add_argument('--cuda_id', type=str, default='3',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--glove_dir', type=str, default='',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='bert-base-uncased',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')

    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')

    parser.add_argument('--add_non_connect', type=bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop', type=bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type=int, default=4,
                        help='max number of hops')

    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')

    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')

    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_mem_dim', type=int, default=300,
                        help='Dimension of the W in GCN.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for GCN.')
    # GAT
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type=str, choices=['linear', 'dotprod', 'gcn'], default='dotprod',
                        help='The attention used for gat')

    parser.add_argument('--embedding_type', type=str, default='glove', choices=['glove', 'bert'])
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')

    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--exp_name', type=str, default='default',
                        help="exp name")
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help="max seq length")
    parser.add_argument('--cross_attn_heads', type=int, default=4,
                        help="cross attention head num")
    parser.add_argument('--use_spc', action='store_true',
                        help="use spc")
    parser.add_argument('--use_cross_attn', type=str2bool, const=True, nargs='?',
                        help="use cross attn")
    parser.add_argument('--use_ner_feature', type=str2bool, const=True, nargs='?',
                        help="use ner feature")
    parser.add_argument('--use_gat_feature', type=str2bool, const=True, nargs='?',
                        help="use ner feature")
    parser.add_argument('--use_bert_global', type=str2bool, const=True, nargs='?',
                        help="use ner feature")
    parser.add_argument('--save_model', type=str2bool, const=True, nargs='?',
                        help="save model")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="loss alpha")
    parser.add_argument('--cache_tag', type=str2bool, const=True, nargs='?',
                        help="")
    parser.add_argument('--eval_badcase', type=str2bool, const=False, nargs='?',
                        help="")
    parser.add_argument('--model_path', type=str, default='/home/cq/RGAT-ABSA/data/output-gcn/default/best_model.pkl',
                        help="model path")
    parser.add_argument('--badcase_output', type=str, default='',
                        help="model path")
    return parser.parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Parse args
    args = parse_args()

    check_args(args)

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer
    args.num_labels = NerUtils.num_labels
    # Load datasets and vocabs
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = load_datasets_and_vocabs(args)
    if args.eval_badcase:
        eval_badcase(args, test_dataset, word_vocab)
        return
        # Build Model
    # model = Aspect_Text_Multi_Syntax_Encoding(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
    if args.pure_bert:
        model = Pure_Bert(args)
    elif args.gat_bert:
        model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT + Bert
    elif args.gat_our:
        model = Aspect_Text_GAT_ours(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT with reshaped tree
    else:
        model = Aspect_Text_GAT_only(args, dep_tag_vocab['len'],
                                     pos_tag_vocab['len'])  # original GAT with reshaped tree

    model.to(args.device)
    # Train
    _, _, all_eval_results = train(args, train_dataset, model, test_dataset)

    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc'])
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))


def eval_badcase(args, dataset, word_vocab):
    logger.info('start eval badcase....')
    model = torch.load(args.model_path)
    from trainer import evaluate_badcase
    import json
    badcases = evaluate_badcase(args, dataset, model, word_vocab)

    def remove_pad(sentence):
        res = []
        for i in sentence.split(' '):
            if i == '[PAD]':
                break
            else:
                res.append(i)
        return ' '.join(res)

    with open(args.badcase_output, 'w', encoding='utf8') as f:
        for case in badcases:
            case['sentence'] = remove_pad(case['sentence'])
            case['aspect'] = remove_pad(case['aspect'])
            f.write(json.dumps(case, ensure_ascii=False) + '\n')


# def error_analysis(path):
#     import json
#     sen = open(path, 'r', encoding='utf8').readlines()[:100]
#     cnt = 0  # 统计中性标签
#     for s in sen:
#         jl = json.loads(s)
#         cnt += 1 if jl['label'] == 2 else 0
#     print(cnt)


if __name__ == "__main__":
    main()
