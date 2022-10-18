from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import string
import os
from pyemd import emd, emd_with_flow
from math import log
from itertools import chain

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
import sys
import os
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from distutils.version import LooseVersion
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
from prettytable import PrettyTable

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn', adapter_path=None):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        if adapter_path:
            self.model.load_adapter(adapter_path,set_active=True)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))





from mt_metrics_eval import data

from json.tool import main
import os
import sys
import time
import pathlib
from weakref import ref
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

from collections import defaultdict

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='facebook/bart-base', type=str)
    parser.add_argument("--adapter_path", default=None, type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default='cuda', type=str)

    args = parser.parse_args()
    return args

   
args = parse_args()
bart_scorer = BARTScorer(device=args.device, checkpoint=args.model_type, adapter_path=args.adapter_path)


def MyMetric(out, ref):
  """Return a scalar score for given output/reference texts."""
  R = bart_scorer.score(ref, out, batch_size=args.batch_size)
  P = bart_scorer.score(out, ref, batch_size=args.batch_size)
  F = [(p+r)/2 for p, r in zip(P,R)]
  return float(sum(F))


test_list = ['cs-en','de-en','iu-en','ja-en','km-en','pl-en','ps-en','ru-en','ta-en','zh-en']


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