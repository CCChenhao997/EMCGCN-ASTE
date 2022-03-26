import numpy as np
from sklearn import metrics
from data import label2id, id2label


def get_aspects(tags, length, token_range, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = id2label[tags[l][l]]
        if label == 'B-A':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-A':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length-1])
    
    return spans


def get_opinions(tags, length, token_range, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l][l] == ignore_index:
            continue
        label = id2label[tags[l][l]]
        if label == 'B-O':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-O':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, length-1])
    
    return spans


class Metric():
    def __init__(self, args, predictions, goldens, bert_lengths, sen_lengths, tokens_ranges, ignore_index=-1):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[3] == 0: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        # label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}
        triplets_utm = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * len(label2id)
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1

                if sum(tag_num[7:]) == 0: continue
                sentiment = -1
                if tag_num[9] >= tag_num[8] and tag_num[9] >= tag_num[7]:
                    sentiment = 9
                elif tag_num[8] >= tag_num[7] and tag_num[8] >= tag_num[9]:
                    sentiment = 8
                elif tag_num[7] >= tag_num[9] and tag_num[7] >= tag_num[8]:
                    sentiment = 7
                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    exit()
                triplets_utm.append([al, ar, pl, pr, sentiment])
        
        return triplets_utm

    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                golden_tuples = self.find_triplet(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))
            
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    
    def score_uniontags_print(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        all_golden_triplets = []
        all_predicted_triplets = []
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                golden_tuples = self.find_triplet(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i])
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))
            
            all_golden_triplets.append(golden_tuples)
            all_predicted_triplets.append(predicted_tuples)

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1, all_golden_triplets, all_predicted_triplets

    def tagReport(self):
        print(len(self.predictions))
        print(len(self.goldens))

        golden_tags = []
        predict_tags = []
        for i in range(self.data_num):
            for r in range(102):
                for c in range(r, 102):
                    if self.goldens[i][r][c] == -1:
                        continue
                    golden_tags.append(self.goldens[i][r][c])
                    predict_tags.append(self.predictions[i][r][c])
        
        print(len(golden_tags))
        print(len(predict_tags))
        target_names = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
        print(metrics.classification_report(golden_tags, predict_tags, target_names=target_names, digits=4))