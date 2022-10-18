from __future__ import absolute_import, division, print_function

import argparse
import os
import string
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool

import numpy as np
import torch
from mt_metrics_eval import data
from prettytable import PrettyTable
from pyemd import emd, emd_with_flow
from scorer import BERTScorer
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='bert-large-uncased', type=str)
    parser.add_argument("--adapter_path", default=None, type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default='cuda',type=str)

    args = parser.parse_args()
    return args


args = parse_args()

scorer = BERTScorer(lang="en", rescale_with_baseline=False, model_type=args.model_type,
                    batch_size=args.batch_size, device=args.device ,adapter_path=args.adapter_path)


def MyMetric(out, ref):
    """Return a scalar score for given output/reference texts."""
    P, R, F = scorer.score(out, ref)
    return float(sum(F))


test_list = ['cs-en', 'de-en', 'iu-en', 'ja-en',
             'km-en', 'pl-en', 'ps-en', 'ru-en', 'ta-en', 'zh-en']


pt = PrettyTable()
ave = 0
count = 0
for test in test_list:
    evs = data.EvalSet('wmt20', test)
    sys_scores, doc_scores, seg_scores = {}, {}, {}
    ref = evs.ref
    for s, out in evs.sys_outputs.items():
        sys_scores[s] = [MyMetric(out, ref)]
    ave += evs.Correlation('sys', sys_scores).Pearson()[0]
    count += 1
    pt.add_column(test, [round(evs.Correlation('sys', sys_scores).Pearson()[0],3)])

pt.add_column('average', [str(round(ave / count, 3))])
print(pt)
