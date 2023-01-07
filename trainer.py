import json
import logging
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from seqeval.metrics import classification_report
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import my_collate, my_collate_elmo, my_collate_pure_bert, my_collate_bert
from transformers import AdamW
from transformers import BertTokenizer

from ner_utils import NerUtils

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {'sentence': batch[0],
                  'aspect': batch[1],  # aspect token
                  'dep_tags': batch[2],  # reshaped
                  'pos_class': batch[3],
                  'text_len': batch[4],
                  'aspect_len': batch[5],
                  'dep_rels': batch[7],  # adj no-reshape
                  'dep_heads': batch[8],
                  'aspect_position': batch[9],
                  'dep_dirs': batch[10]
                  }
        labels = batch[6]
    else:  # bert
        if args.pure_bert:
            # input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[1]}
            labels = batch[6]
        else:
            # input_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {'input_ids': batch[0],
                      'input_aspect_ids': batch[2],
                      'word_indexer': batch[1],
                      'aspect_indexer': batch[3],
                      'input_cat_ids': batch[4],
                      'segment_ids': batch[5],
                      'dep_tags': batch[6],
                      'pos_class': batch[7],
                      'text_len': batch[8],
                      'aspect_len': batch[9],
                      'dep_rels': batch[11],
                      'dep_heads': batch[12],
                      'aspect_position': batch[13],
                      'dep_dirs': batch[14],
                      'ner_valid': batch[16],
                      'ner_mask': batch[17],
                      'ner_input_mask': batch[18],
                      'ner_segment_ids': batch[19]
                      }
            labels = batch[10]
            ner_labels = batch[15]
    return inputs, labels, ner_labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        if args.pure_bert:
            return my_collate_pure_bert
        else:
            return my_collate_bert


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer


def train(args, train_dataset, model, test_dataset):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    args.best_result = {'acc': 0, 'f1': 0, 'ate_f1': 0}
    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # 下面的label应该只有一个值，用于情感分类
            inputs, labels, ner_labels = get_input_from_batch(args, batch)
            logit, ner_logit = model(**inputs)
            loss_cls = F.cross_entropy(logit, labels)
            # 加一个序列标注任务，多任务联合训练
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss_ner = loss_fct(ner_logit.view(-1, 6), ner_labels.view(-1))
            # 之前的实验都是基于1比1的loss，后面加上比例系数alpha
            # loss = 1 * loss_cls + 1 * loss_ner
            loss = args.alpha * loss_cls + (1 - args.alpha) * loss_ner
            # # todo：删掉下面的loss，用于单个ate任务
            # loss = loss_ner
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, test_dataset, model)
                    all_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step, all_eval_results


def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ner_true = []
    ner_pred = []
    for batch in eval_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels, ner_labels = get_input_from_batch(args, batch)

            logits, ner_logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
        y_true, y_pred = cal_ner_metric(ner_logits, ner_labels)
        ner_true.extend(y_true)
        ner_pred.extend(y_pred)
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    result_ner = classification_report(ner_true, ner_pred, digits=4)
    ate_f1 = round(float(result_ner.split()[7]) * 100, 2)
    result['ate_f1'] = ate_f1
    results.update(result)
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    output_eval_file = os.path.join(output_dir, 'eval_results.txt')
    best_eval_file = os.path.join(output_dir, 'eval_best_results.txt')
    if result['acc'] > args.best_result['acc'] and args.save_model:
        # 保存模型
        torch.save(model, os.path.join(output_dir, 'best_model.pkl'))
    args.best_result['acc'] = max(args.best_result['acc'], result['acc'])
    args.best_result['f1'] = max(args.best_result['f1'], result['f1'])
    args.best_result['ate_f1'] = max(args.best_result['ate_f1'], result['ate_f1'])
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** Eval results *****')
        logger.info("  eval loss: %s", str(eval_loss))
        for key in sorted(result.keys()):
            logger.info("  %s = %s (max : %s)", key, str(result[key]), args.best_result[key])
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    if result['acc'] >= args.best_result['acc']:
        with open(best_eval_file, 'w') as f:
            f.write('***** max results*****\n')
            f.write(json.dumps(args.best_result, ensure_ascii=False) + '\n')
            f.write('***** best results*****\n')
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        # f.write('***** ner results*****\n')
        # f.write(result_ner + '\n')
    return results, eval_loss


def cal_ner_metric(ner_logits, ner_labels):
    """
    计算ner的相关指标
    """
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(NerUtils.label_list, 1)}
    logits = torch.argmax(F.log_softmax(ner_logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()
    label_ids = ner_labels.to('cpu').numpy()
    for i, label in enumerate(label_ids):
        temp_1 = []
        temp_2 = []
        for j, m in enumerate(label):
            if j == 0:
                continue
            elif label_ids[i][j] == len(label_map):
                y_true.append(temp_1)
                y_pred.append(temp_2)
                if logits[i][j] != len(label_map):
                    print(temp_1)
                    print(temp_2)
                break
            else:
                temp_1.append(label_map.get(label_ids[i][j], 'O'))
                temp_2.append(label_map.get(logits[i][j], 'O'))
    return y_true, y_pred


def evaluate_badcase(args, eval_dataset, model, word_vocab):
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    badcases = []
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels, ner_labels = get_input_from_batch(args, batch)

            logits, ner_labels = model(**inputs)
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()
        labels = labels.detach().cpu().numpy().tolist()
        for i in range(len(preds)):
            pred = int(preds[i])
            label = int(labels[i])
        # pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        # label = int(labels.detach().cpu().numpy()[0])
            if pred != label:
                if args.embedding_type == 'bert':
                    sent_ids = inputs['input_ids'][i].detach().cpu().numpy()
                    aspect_ids = inputs['input_aspect_ids'][i].detach().cpu().numpy()
                    case = {}
                    case['sentence'] = args.tokenizer.decode(sent_ids)
                    case['aspect'] = args.tokenizer.decode(aspect_ids)
                    case['pred'] = pred
                    case['label'] = label
                    badcases.append(case)
                else:
                    sent_ids = inputs['sentence'][0].detach().cpu().numpy()
                    aspect_ids = inputs['aspect'][0].detach().cpu().numpy()
                    case = {}
                    case['sentence'] = ' '.join([word_vocab['itos'][i] for i in sent_ids])
                    case['aspect'] = ' '.join([word_vocab['itos'][i] for i in aspect_ids])
                    case['pred'] = pred
                    case['label'] = label
                    badcases.append(case)

    return badcases


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": round(acc * 100, 2),
        "f1": round(f1 * 100, 2)
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)
