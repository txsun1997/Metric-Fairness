import argparse
import random
from math import fabs

import fitlog
import numpy as np
import torch
import torch.nn as nn
from dataloader import DataLoader
from fastNLP import (AccuracyMetric, ClassifyFPreRecMetric, DataSet,
                     FitlogCallback, GradientClipCallback, Instance,
                     LossInForward, RandomSampler, Tester, Trainer,
                     WarmupCallback, cache_results)
from transformers import AdamW, BartForConditionalGeneration, BartTokenizer


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", default='facebook/bart-base', type=str, required=False)
    parser.add_argument(
        "--adapter_name", default='debiased-bartscore', type=str, required=False)
    parser.add_argument("--lr", default=1e-3, type=float, required=False)
    parser.add_argument("--warmup", default=0.0, type=float, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--n_epochs", default=4, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=100,
                        type=int, required=False)
    parser.add_argument("--bart_batch_size", default=8,
                        type=int, required=False)
    parser.add_argument("--max_length", default=1024, type=int, required=False)
    parser.add_argument(
        "--data_path", default='train.tsv', type=str, required=False)
    return parser.parse_args()


class BARTScore(torch.nn.Module):
    def __init__(self, args):
        super(BARTScore, self).__init__()

        self.tokenizer = BartTokenizer.from_pretrained(args.model_type)
        self.model = BartForConditionalGeneration.from_pretrained(
            args.model_type)
        # print(self.model)
        # print(type(self.model))
        self.model.add_adapter(args.adapter_name)
        # add adapter and freeze other parameters
        self.model.train_adapter(args.adapter_name)
        self.model.to(args.device)

        self.loss_fct = nn.NLLLoss(
            reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)
        self.batch_size = args.bart_batch_size
        self.device = args.device
        self.max_length = args.max_length

    def save_adapter(self, adapter_name):
        self.model.save_adapter('./adapter', adapter_name)

    def get_bart_score(self, src, tgt):
        for i in range(0, len(src), self.batch_size):
            src_list = src[i: i + self.batch_size]
            tgt_list = tgt[i: i + self.batch_size]

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
            tgt_mask = encoded_tgt['attention_mask'].to(self.device)
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
            if i == 0:
                score = -loss
            else:
                score = torch.cat((score, -loss), 0)
        return score

    def forward(self, refs, hyps, labels):
        refs = refs.tolist()
        hyps = hyps.tolist()
        r = self.get_bart_score(hyps, refs)
        p = self.get_bart_score(refs, hyps)
        f = (r + p) / 2
        loss_func = torch.nn.MSELoss()
        loss = loss_func(f, labels)
        return {
            'p': p,
            'r': r,
            'f': f,
            'loss': loss,
        }


if __name__ == '__main__':
    args = parse_args()
    set_seed(args)

    # static hyperparams
    args.all_layers = False
    args.lang = 'en'
    args.verbose = False
    args.adapter_name = args.model_type + args.adapter_name

    log_dir = './logs'
    fitlog.set_log_dir(log_dir)
    # fitlog.commit(__file__)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    model = BARTScore(args)

    @cache_results('cached_data.bin', _refresh=False)
    def get_data(path):
        paths = {
            'train': path,
        }
        data_bundle = DataLoader().load(paths)
        return data_bundle

    # load dataset
    data_bundle = get_data(path=args.data_path)
    train_data = data_bundle.get_dataset('train')
    print('# samples: {}'.format(len(train_data)))
    print('Example:')
    print(train_data[0])

    parameters = []
    # print('Trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(param)
            # print('{}: {}'.format(name, param.shape))
    optimizer = AdamW(parameters, lr=args.lr)

    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=1, clip_type='norm'))
    callbacks.append(FitlogCallback(log_loss_every=args.logging_steps))
    if args.warmup > 0:
        callbacks.append(WarmupCallback(warmup=args.warmup, schedule='linear'))
    trainer = Trainer(train_data=train_data, model=model, loss=LossInForward(), optimizer=optimizer,
                      batch_size=args.batch_size, sampler=RandomSampler(), drop_last=False, update_every=1,
                      num_workers=4, n_epochs=args.n_epochs, print_every=50, dev_data=None, metrics=None,
                      validate_every=args.logging_steps, save_path=None, use_tqdm=False, device=args.device,
                      callbacks=callbacks, dev_batch_size=None, metric_key=None)
    trainer.train(load_best_model=False)
    model.save_adapter(args.adapter_name)
    fitlog.finish()
