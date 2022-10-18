import argparse
import random
from collections import defaultdict

import fitlog
import numpy as np
import pandas as pd
import torch
from bertscore_utils import (bert_cos_score_idf, cache_scibert,
                             get_bert_embedding, get_idf_dict, get_model,
                             get_tokenizer, lang2model, model2layers,
                             sent_encode)
from dataloader import DataLoader
from fastNLP import (FitlogCallback, GradientClipCallback, LossInForward,
                     RandomSampler, Trainer, WarmupCallback, cache_results)
from transformers import AdamW


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", default='bert-large-uncased', type=str, required=False)
    parser.add_argument(
        "--adapter_name", default='debiased-bertscore', type=str, required=False)
    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--warmup", default=0.0, type=float, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--n_epochs", default=4, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=100,
                        type=int, required=False)
    parser.add_argument(
        "--data_path", default='train.tsv', type=str, required=False)
    return parser.parse_args()


class BERTScore(torch.nn.Module):
    def __init__(self, args):
        super(BERTScore, self).__init__()
        num_layers = model2layers[args.model_type]
        self.tokenizer = get_tokenizer(args.model_type)
        self.model = get_model(args.model_type, num_layers, all_layers=True)
        self.model.add_adapter(args.adapter_name)
        # add adapter and freeze other parameters
        self.model.train_adapter(args.adapter_name)
        self.model.to(args.device)

        self.idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0

        self.verbose = args.verbose
        self.device = args.device
        self.batch_size = args.batch_size
        self.all_layers = args.all_layers

    def save_adapter(self, adapter_name):
        self.model.save_adapter('./adapter', adapter_name)

    def forward(self, refs, hyps, labels):
        refs = refs.tolist()
        hyps = hyps.tolist()
        all_preds = bert_cos_score_idf(
            self.model,
            refs,
            hyps,
            self.tokenizer,
            self.idf_dict,
            verbose=self.verbose,
            device=self.device,
            batch_size=self.batch_size,
            all_layers=self.all_layers,
        )

        p, r, f = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]

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

    model = BERTScore(args)

    @cache_results('cached_data.bin', _refresh=False)
    def get_data(path):
        paths = {
            'train': path,
        }
        data_bundle = DataLoader().load(paths)
        return data_bundle

    # load dataset
    data_bundle = get_data(patt=args.data_path)
    train_data = data_bundle.get_dataset('train')
    print('# samples: {}'.format(len(train_data)))
    print('Example:')
    print(train_data[0])

    parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(param)
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
