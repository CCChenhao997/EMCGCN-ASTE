#coding utf-8

import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator, label2id
from model import EMCGCN
import utils

import numpy as np

from prepare_vocab import VocabHelp
from transformers import AdamW


def get_bert_optimizer(model, args):
    # # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ["bert.embeddings", "bert.encoder"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    return optimizer


def train(args):

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))

    post_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_post.vocab')
    deprel_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_deprel.vocab')
    postag_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_postag.vocab')
    synpost_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_synpost.vocab')
    args.post_size = len(post_vocab)
    args.deprel_size = len(deprel_vocab)
    args.postag_size = len(postag_vocab)
    args.synpost_size = len(synpost_vocab)

    instances_train = load_data_instances(train_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
    instances_dev = load_data_instances(dev_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = EMCGCN(args).to(args.device)
    optimizer = get_bert_optimizer(model, args)

    # label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).float().cuda()

    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(args.epochs):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            _, sentences, tokens, lengths, masks, _, _, aspect_tags, tags, word_pair_position, \
            word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry = trainset.get_batch(j)
            tags_flatten = tags.reshape([-1])
            tags_symmetry_flatten = tags_symmetry.reshape([-1])
            if args.relation_constraint:
                predictions = model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost)
                biaffine_pred, post_pred, deprel_pred, postag, synpost, final_pred = predictions[0], predictions[1], predictions[2], predictions[3], predictions[4], predictions[5]
                l_ba = 0.10 * F.cross_entropy(biaffine_pred.reshape([-1, biaffine_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_rpd = 0.01 * F.cross_entropy(post_pred.reshape([-1, post_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_dep = 0.01 * F.cross_entropy(deprel_pred.reshape([-1, deprel_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_psc = 0.01 * F.cross_entropy(postag.reshape([-1, postag.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_tbd = 0.01 * F.cross_entropy(synpost.reshape([-1, synpost.shape[3]]), tags_symmetry_flatten, ignore_index=-1)

                if args.symmetry_decoding:
                    l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_symmetry_flatten, weight=weight, ignore_index=-1)
                else:
                    l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_flatten, weight=weight, ignore_index=-1)

                loss = l_ba + l_rpd + l_dep + l_psc + l_tbd + l_p
            else:
                preds = model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost)[-1]
                preds_flatten = preds.reshape([-1, preds.shape[3]])
                if args.symmetry_decoding:
                    loss = F.cross_entropy(preds_flatten, tags_symmetry_flatten, weight=weight, ignore_index=-1)
                else:
                    loss = F.cross_entropy(preds_flatten, tags_flatten, weight=weight, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + 'bert' + args.task + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))


def eval(model, dataset, args, FLAG=False):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_sentences = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        for i in range(dataset.batch_count):
            sentence_ids, sentences, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, \
            word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry = dataset.get_batch(i)
            preds = model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost)[-1]
            preds = F.softmax(preds, dim=-1)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)
            all_ids.extend(sentence_ids)
            all_sentences.extend(sentences)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\t\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

        if FLAG:
            metric.tagReport()

    model.train()
    return precision, recall, f1


def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + 'bert' + args.task + '.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    post_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_post.vocab')
    deprel_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_deprel.vocab')
    postag_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_postag.vocab')
    synpost_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_synpost.vocab')
    instances = load_data_instances(sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args, False)


if __name__ == '__main__':
    # torch.set_printoptions(precision=None, threshold=float("inf"), edgeitems=None, linewidth=None, profile=None)
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../data/D1/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=102,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')
    
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=len(label2id),
                        help='label number')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")

    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--pooling', default='avg', type=str, help='[max, avg, sum]')
    parser.add_argument('--gcn_dim', type=int, default=300, help='dimension of GCN')
    parser.add_argument('--relation_constraint', default=True, action='store_true')
    parser.add_argument('--symmetry_decoding', default=False, action='store_true')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.task == 'triplet':
        args.class_num = len(label2id)

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
