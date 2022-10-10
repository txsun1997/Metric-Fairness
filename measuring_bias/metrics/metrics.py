from bert_score import score
import argparse
from BARTScorer import BARTScorer
from MoverScorer import MoverScorer
from Bleurt import Bleurt
from datasets import load_metric
from rouge_score import rouge_scorer
from nltk.translate.chrf_score import sentence_chrf
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import SmoothingFunction
from prism import Prism
from nltk.corpus import wordnet
import pandas as pd
nltk.download('wordnet')
nltk.download('omw-1.4')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyps_file", default='hyps.txt', type=str)
    parser.add_argument("--refs_file", default='refs.txt', type=str)
    parser.add_argument("--bert_score_model",
                        default='roberta-large', type=str)
    parser.add_argument("--bart_score_model",
                        default='facebook/bart-large-cnn', type=str)
    parser.add_argument("--mover_score_model",
                        default='distilbert-base-uncased', type=str)
    parser.add_argument("--frugal_score_model",
                        default='moussaKam/frugalscore_tiny_bert-base_bert-score', type=str)
    parser.add_argument("--bleurt_score_model",
                        default='Elron/bleurt-base-512', type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--output_file", default='scores.csv', type=str)

    args = parser.parse_args()
    return args


args = parse_args()

with open(args.hyps_file) as f:
    cands = [line.strip() for line in f]

with open(args.refs_file) as f:
    refs = [line.strip() for line in f]


def get_bert_score(args):

    P, R, F1 = score(cands, refs, lang='en',  model_type=args.bert_score_model,
                     device=args.device, batch_size=args.batch_size)

    return {'P': P.tolist(), 'R': R.tolist(), 'F': F1.tolist()}


def get_bart_score(args):

    bart_scorer = BARTScorer(
        device=args.device, checkpoint=args.bart_score_model)

    R = bart_scorer.score(cands, refs, args.batch_size)
    P = bart_scorer.score(refs, cands, args.batch_size)
    F = [(r+p)/2 for r, p in zip(R, P)]
    return {'P': P, 'R': R, 'F': F}


def get_mover_score(args):

    mover_scorer = MoverScorer(args.mover_score_model)
    # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_hyp = mover_scorer.get_idf_dict(cands)
    # idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_ref = mover_scorer.get_idf_dict(refs)

    scores = mover_scorer.word_mover_score(refs, cands, idf_dict_ref, idf_dict_hyp,
                                           stop_words=[], n_gram=2, remove_subwords=True, batch_size=args.batch_size)

    return scores


def get_frugal_score(args):

    metric = load_metric('frugalscore.py')

    scores = metric.compute(references=refs,
                            predictions=cands,
                            pretrained_model_name_or_path=args.frugal_score_model,
                            batch_size=args.batch_size)
    return scores['scores']


def get_bleurt_score(args):

    bleurt = Bleurt(args.bleurt_score_model, args.device)
    scores = []
    for i in range(0, len(cands), args.batch_size):
        cands_batch = cands[i:i+args.batch_size]
        refs_batch = refs[i:i+args.batch_size]
        score = bleurt(refs_batch, cands_batch)['score'].tolist()
        if args.batch_size > 1:
            scores.extend(score)
        else:
            scores.append(score)

    return scores


def get_bleu_score(args):

    scores = []
    for i in range(len(cands)):
        reference = [refs[i].split()]
        candidate = cands[i].split()
        smooth = SmoothingFunction()
        score = sentence_bleu(reference, candidate, weights=(
            0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        scores.append(score)
    return scores


def get_rouge_score(args):

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = []
    for i in range(len(cands)):
        score = scorer.score(refs[i], cands[i])
        scores.append(score['rouge1'][2])
    return scores


def get_nist_score(args):
    scores = []
    for i in range(len(cands)):
        reference = [refs[i].split()]
        candidate = cands[i].split()

        score = sentence_nist(reference, candidate)
        scores.append(score)
    return scores


def get_chrf_score(args):

    scores = []
    for i in range(len(cands)):
        reference = refs[i]
        candidate = cands[i]
        score = sentence_chrf(reference, candidate)
        scores.append(score)
    return scores


def get_meteor_score(args):

    scores = []
    for i in range(len(cands)):
        hypothesis = cands[i].split()
        reference = refs[i].split()
        hypothesis = set(hypothesis)
        reference = set(reference)

        merteor_score = nltk.translate.meteor_score.single_meteor_score(
            reference, hypothesis)
        scores.append(merteor_score)
    return scores


def get_prism_score(args):
    prism = Prism(model_dir='m39v1', lang='en')
    return prism.score(cand=cands, ref=refs, segment_scores=True)


METRIC_TO_FUNC = {
    'bleu': get_bleu_score,
    'rouge': get_rouge_score,
    'meteor': get_meteor_score,
    'nist': get_nist_score,
    'chrf': get_chrf_score,
    'bertscore': get_bert_score,
    'moverscore': get_mover_score,
    'bleurt': get_bleurt_score,
    'prism': get_prism_score,
    'bartscore': get_bart_score,
    'frugalscore': get_frugal_score
}


def print_result(args):
    score_dict = {'candidates': cands, 'reference': refs}
    for key in METRIC_TO_FUNC.keys():
        print('starting calculate ' + key)
        func = METRIC_TO_FUNC[key]
        if key.startswith('bertscore') or key.startswith('bartscore') or key.startswith('prism'):

            score_dict[key+'_r'] = func(args)['R']
            score_dict[key+'_p'] = func(args)['P']
            score_dict[key+'_f'] = func(args)['F']
        else:
            score_dict[key] = func(args)

    df = pd.DataFrame(score_dict)
    df.to_csv(args.output_file)


if __name__ == '__main__':
    print_result(args)
