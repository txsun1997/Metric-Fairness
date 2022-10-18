import sys
from sys import path
from prettytable import PrettyTable
import argparse
path.append('BERTScore')
path.append('BLEURT')
path.append('BARTScore')


from test_bert_score import cal_bert_score
from test_bleurt import cal_bleurt
from test_bart_score import cal_bart_score


with open('test_data/hyps.txt') as f:
    cands = [line.strip() for line in f]

with open('test_data/refs.txt') as f:
    refs = [line.strip() for line in f]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_score_bert_large_adapter_path", default='BERTScore/BERT-large/adapter', type=str)
    parser.add_argument("--bert_score_bert_base_adapter_path", default='BERTScore/BERT-base/adapter', type=str)
    parser.add_argument("--bleurt_bert_base_adapter_path", default='BLEURT/adapter', type=str)
    parser.add_argument("--bart_score_bart_base_adapter_path", default='BARTScore/adapter', type=str)
    args = parser.parse_args()
    return args


def cal_bias_score(data):
    max_value = max(data)
    min_value = min(data)
    normalized_data = []
    for x in data:
        tmp = (x - min_value) / (max_value - min_value) * 100
        normalized_data.append(tmp)

    import math
    avg_delta = []
    tmp = None
    for x in normalized_data:
        if tmp is None:
            tmp = x
        else:
            tmp = math.fabs(tmp - x)
            avg_delta.append(tmp)
            tmp = None
    return round(sum(avg_delta) / len(avg_delta),2)

args = parse_args()


bert_score_bert_base = cal_bert_score(cands, refs, 'bert-base-uncased',args.bert_score_bert_base_adapter_path)
bert_score_bert_large = cal_bert_score(cands, refs, 'bert-large-uncased',args.bert_score_bert_large_adapter_path)
bleurt = cal_bleurt(cands,refs,adapter_path=args.bleurt_bert_base_adapter_path)
bart_score = cal_bart_score(cands,refs,adapter_path=args.bart_score_bart_base_adapter_path)


pt = PrettyTable()
pt.add_column('bert_score_bert_base',[cal_bias_score(bert_score_bert_base)])
pt.add_column('bert_score_bert_large',[cal_bias_score(bert_score_bert_large)])
pt.add_column('bleurt_bert_base',[cal_bias_score(bleurt)])
pt.add_column('bart_score_bart_base',[cal_bias_score(bart_score)])

print(pt)