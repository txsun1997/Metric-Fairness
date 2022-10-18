import argparse
from ast import arg
from collections import defaultdict

import fitlog
import numpy as np
import torch
from dataloader import DataLoader
from fastNLP import (AccuracyMetric, ClassifyFPreRecMetric, DataSet,
                     FitlogCallback, GradientClipCallback, Instance,
                     LossInForward, RandomSampler, Tester, Trainer,
                     WarmupCallback, cache_results)
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer)


def padding(squence, max_length, pad_token=0):
    padding_length = max_length - len(squence)
    return squence + [pad_token] * padding_length


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default='Elron/bleurt-base-512', type=str, required=False)
    parser.add_argument("--adapter_name", default='debiased-bleurt', type=str, required=False)
    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--warmup", default=0.0, type=float, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--n_epochs", default=6, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=100, type=int, required=False)
    parser.add_argument(
        "--data_path", default='/remote-home/share/jlhe/train_Bleurt_base_single.tsv', type=str, required=False)

    return parser.parse_args()





class Bleurt(torch.nn.Module):
    def __init__(self, args):
        super(Bleurt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_type)
        self.model.add_adapter(args.adapter_name)
        self.model.train_adapter(args.adapter_name)
        self.model.to(args.device)
        self.device = args.device

    def save_adapter(self, adapter_name):
        self.model.save_adapter('./adapter',adapter_name)

    def forward(self, refs, hyps, labels):
        refs = refs.tolist()
        hyps = hyps.tolist()
        tokens = self.tokenizer(refs, hyps,add_special_tokens=True, max_length=512,  
        truncation=True)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        token_type_ids = tokens['token_type_ids']


        for i in range(len(input_ids)):
            input_ids[i] = padding(input_ids[i], 512, pad_token=0)
            attention_mask[i] = padding(attention_mask[i], 512, pad_token=0)
            token_type_ids[i] = padding(token_type_ids[i], 512, pad_token=0)

        input_ids = torch.tensor(input_ids,device=self.device)
        attention_mask = torch.tensor(attention_mask,device=self.device)
        token_type_ids = torch.tensor(token_type_ids,device=self.device)
        scores = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0].squeeze()
        loss_func = torch.nn.MSELoss()
        loss = loss_func(scores, labels)


        return {
            'score': scores,
            'loss': loss,
        }









import random

if __name__ == '__main__':
    args = parse_args()
    set_seed(args)
    args.adapter_name = args.model_type + args.adapter_name
    log_dir = './logs'
    fitlog.set_log_dir(log_dir)
    # fitlog.commit(__file__)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    model = Bleurt(args)

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
                      num_workers=1, n_epochs=args.n_epochs, print_every=50, dev_data=None, metrics=None,
                      validate_every=args.logging_steps, save_path=None, use_tqdm=False, device=args.device,
                      callbacks=callbacks, dev_batch_size=None, metric_key=None)
    trainer.train(load_best_model=False)
    model.save_adapter(args.adapter_name)
    fitlog.finish()
