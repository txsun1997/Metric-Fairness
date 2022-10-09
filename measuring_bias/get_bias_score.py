import os
import sys
import argparse
from prettytable import PrettyTable


VALID_BIAS_TYPES = [
    'age',
    'gender',
    'race',
    'religion',
    'socioeconomic',
    'physical-appearance',
]

VALID_METRIC_NAME = [
    'bleu',
    'rouge',
    'meteor',
    'nist',
    'chrf',
    'bertscore-distilbert',
    'bertscore-roberta-base',
    'bertscore-roberta-large',
    'bertscore-bert-base',
    'bertscore-bert-large',
    'moverscore-distilbert',
    'moverscore-bert-base',
    'moverscore-bert-large',
    'bleurt-bert-tiny',
    'bleurt-bert-base',
    'bleurt-bert-large',
    'bleurt-rembert',
    'prism-p',
    'prism-r',
    'prism-f',
    'bartscore-bart-base-p',
    'bartscore-bart-base-r',
    'bartscore-bart-base-f',
    'bartscore-bart-large-p',
    'bartscore-bart-large-r',
    'bartscore-bart-large-f',
    'frugalscore-bert-tiny',
    'frugalscore-bert-small',
    'frugalscore-bert-medium',
]


def print_results(scores, metric_names):
    bias_types = sorted(scores.keys())
    table = PrettyTable(['metric'] + bias_types)
    for mn in sorted(metric_names):
        row = [mn]
        for bt in bias_types:
            if mn in scores[bt]:
                row.append(str(round(scores[bt][mn], 2)))
            else:
                row.append('-')
        table.add_row(row)
    print(table)


def compute_bias_score(scores):
    # step 1: normalize scores
    normalized_scores = []
    s_min = min(scores)
    s_max = max(scores)
    for s in scores:
        normalized_scores.append(100 * (s-s_min)/(s_max-s_min))

    # step 2: compute absolute difference
    bias_score = 0.0
    s0 = None
    for s in normalized_scores:
        if s0 is None:
            s0 = s
        else:
            bias_score += abs(s-s0)
            s0 = None
    bias_score = 2 * bias_score / len(normalized_scores)

    return bias_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias_type", default=['all'], nargs='+', type=str)
    parser.add_argument("--metric_name", default=['all'], nargs='+', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.bias_type[0] == 'all':
        eval_bias_types = VALID_BIAS_TYPES
    else:
        # check if input bias types valid
        eval_bias_types = []
        for bt in args.bias_type:
            if bt in VALID_BIAS_TYPES:
                eval_bias_types.append(bt)
            else:
                print('Invalid bias type: {}. Valid bias types: {}'.format(
                    bt, VALID_BIAS_TYPES))

    print('Evaluating bias types: {}'.format(eval_bias_types))

    if args.metric_name[0] == 'all':
        eval_metric_names = VALID_METRIC_NAME
    else:
        # check if input metric names valid
        eval_metric_names = []
        for mn in args.metric_name:
            if mn in VALID_METRIC_NAME:
                eval_metric_names.append(mn)
            else:
                print('Invalid metric name: {}. Valid metric names: {}'.format(
                    mn, VALID_METRIC_NAME))

    print('Evaluating metric names: {}'.format(eval_metric_names))

    # begin evaluating
    overall_scores = {}
    data_paths = [os.path.join('./data', eval_bias_types[i]+'.tsv')
                  for i in range(len(eval_bias_types))]
    for bias_type, data_dir in zip(eval_bias_types, data_paths):
        if not os.path.exists(data_dir):
            print('{} does not exist. Please check your data.'.format(data_dir))
            continue
        with open(data_dir, 'r', encoding='utf-8') as f:
            print('\nReading {}...'.format(data_dir))
            lines = f.readlines()
            num_sample_pairs = int((len(lines) - 1) / 2)
            print('# sample pairs: {}'.format(num_sample_pairs))

            items = lines[0].strip().split('\t')
            # print(items)
            heads = {}
            for col, item in enumerate(items):
                if item.lower() in eval_metric_names:
                    heads[item.lower()] = col

            # check if we miss some metric names
            for mn in eval_metric_names:
                if mn not in heads.keys():
                    print('{} not found.'.format(mn))

            bias_scores = {}
            for line in lines[1:]:
                if len(line) > 1:
                    vals = line.strip().split('\t')
                    assert len(vals) == len(items)
                    for mn, col in heads.items():
                        if mn not in bias_scores:
                            bias_scores[mn] = [float(vals[col])]
                        else:
                            bias_scores[mn].append(float(vals[col]))

            for mn, metric_scores in bias_scores.items():
                bias_scores[mn] = compute_bias_score(metric_scores)

            overall_scores[bias_type] = bias_scores
            
    # print results
    print('\nTable: Evaluated bias scores.')
    print_results(overall_scores, eval_metric_names)
