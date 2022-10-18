from __future__ import absolute_import, division, print_function
import torch



from mt_metrics_eval import data

from transformers import AutoModelForSequenceClassification, AutoTokenizer,AdamW

from prettytable import PrettyTable
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='Elron/bleurt-base-512', type=str)
    parser.add_argument("--adapter_path", default=None, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default='cuda', type=str)

    args = parser.parse_args()
    return args

args = parse_args()


def padding(squence, max_length, pad_token=0):
    padding_length = max_length - len(squence)
    return squence + [pad_token] * padding_length

class Bleurt(torch.nn.Module):
    def __init__(self,args):
        super(Bleurt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_type)
        if args.adapter_path:
          self.model.load_adapter(args.adapter_path,set_active=True)
        self.model.to(args.device)
        self.model.eval()
        self.device = args.device

    def forward(self, refs, hyps):
        tokens = self.tokenizer(refs, hyps,add_special_tokens=True, max_length=512, truncation=True)
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
        return scores

model = Bleurt(args)


def MyMetric(out, ref):
  """Return a scalar score for given output/reference texts."""
  scores = []
  with torch.no_grad():
    for i in range(0,len(out),args.batch_size):
        cands_batch = out[i:i+args.batch_size]
        refs_batch = ref[i:i+args.batch_size]
        for j in model.forward(refs_batch,cands_batch):
            scores.append(float(j))
    
  return float(sum(scores))


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
