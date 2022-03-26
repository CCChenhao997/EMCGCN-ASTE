"""
Prepare vocabulary and initial word vectors.
"""
import json
import tqdm
import pickle
import argparse
import numpy as np
from collections import Counter
from collections import defaultdict


class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]
        
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)    # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab.')
    parser.add_argument('--data_dir', default='../data/D1/res16', help='data directory.')
    parser.add_argument('--vocab_dir', default='../data/D1/res16', help='Output vocab directory.')
    parser.add_argument('--lower', default=False, help='If specified, lowercase all words.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'

    # output files
    # token
    vocab_tok_file = args.vocab_dir + '/vocab_tok.vocab'
    # position
    vocab_post_file = args.vocab_dir + '/vocab_post.vocab'
    # deprel
    vocab_deprel_file = args.vocab_dir + '/vocab_deprel.vocab'
    # postag
    vocab_postag_file = args.vocab_dir + '/vocab_postag.vocab'
    # syn_post
    vocab_synpost_file = args.vocab_dir + '/vocab_synpost.vocab'

    # load files
    print("loading files...")
    train_tokens, train_deprel, train_postag, train_postag_ca, train_max_len = load_tokens(train_file)
    dev_tokens, dev_deprel, dev_postag, dev_postag_ca, dev_max_len = load_tokens(dev_file)
    test_tokens, test_deprel, test_postag, test_postag_ca, test_max_len = load_tokens(test_file)

    # lower tokens
    if args.lower:
        train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
                (train_tokens, dev_tokens, test_tokens)]

    # counters
    token_counter = Counter(train_tokens + dev_tokens + test_tokens)
    deprel_counter = Counter(train_deprel + dev_deprel + test_deprel)
    postag_counter = Counter(train_postag + dev_postag + test_postag)
    postag_ca_counter = Counter(train_postag_ca + dev_postag_ca + test_postag_ca)
    # deprel_counter['ROOT'] = 1
    deprel_counter['self'] = 1
    
    max_len = max(train_max_len, dev_max_len, test_max_len)
    # post_counter = Counter(list(range(-max_len, max_len)))
    post_counter = Counter(list(range(0, max_len)))
    syn_post_counter = Counter(list(range(0, 5)))

    # build vocab
    print("building vocab...")
    token_vocab  = VocabHelp(token_counter, specials=['<pad>', '<unk>'])
    post_vocab   = VocabHelp(post_counter, specials=['<pad>', '<unk>'])
    deprel_vocab = VocabHelp(deprel_counter, specials=['<pad>', '<unk>'])
    postag_vocab = VocabHelp(postag_counter, specials=['<pad>', '<unk>'])
    syn_post_vocab = VocabHelp(syn_post_counter, specials=['<pad>', '<unk>'])
    print("token_vocab: {}, post_vocab: {}, syn_post_vocab: {}, deprel_vocab: {}, postag_vocab: {}".format(len(token_vocab), len(post_vocab), len(syn_post_vocab), len(deprel_vocab), len(postag_vocab)))

    print("dumping to files...")
    # token_vocab.save_vocab(vocab_tok_file)
    post_vocab.save_vocab(vocab_post_file)
    deprel_vocab.save_vocab(vocab_deprel_file)
    postag_vocab.save_vocab(vocab_postag_file)
    syn_post_vocab.save_vocab(vocab_synpost_file)
    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        deprel = []
        postag = []
        postag_ca = []
        
        max_len = 0
        for d in data:
            sentence = d['sentence'].split()
            tokens.extend(sentence)
            deprel.extend(d['deprel'])
            postag_ca.extend(d['postag'])
            # postag.extend(d['postag'])
            n = len(d['postag'])
            tmp_pos = []
            for i in range(n):
                for j in range(n):
                    tup = tuple(sorted([d['postag'][i], d['postag'][j]]))
                    tmp_pos.append(tup)
            postag.extend(tmp_pos)
            
            max_len = max(len(sentence), max_len)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, deprel, postag, postag_ca, max_len


if __name__ == '__main__':
    main()