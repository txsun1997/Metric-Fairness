import sys
from sys import path
path.append('test/BERTScore/bert_score')
from scorer import BERTScorer




def cal_bert_score(out, ref, model_tpye, adapter_path=None, batch_size=8):
    scorer = BERTScorer(lang="en", rescale_with_baseline=False,
                        model_type=model_tpye, batch_size=batch_size, adapter_path=adapter_path)
    P, R, F = scorer.score(out, ref)
    return F.tolist()


