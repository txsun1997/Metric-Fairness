import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer,AdamW






def padding(squence, max_length, pad_token=0):
    padding_length = max_length - len(squence)
    return squence + [pad_token] * padding_length

class Bleurt(torch.nn.Module):
    def __init__(self,adapter_path,device,model_type):
        super(Bleurt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_type)
        if adapter_path:
            self.model.load_adapter(adapter_path,set_active=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

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




def cal_bleurt(out, ref, adapter_path=None,device='cuda',model_type='Elron/bleurt-base-512'):
  model = Bleurt(adapter_path,device,model_type)
  scores = []
  with torch.no_grad():
    for i in range(0,len(out),128):
        cands_batch = out[i:i+128]
        refs_batch = ref[i:i+128]
        for j in model.forward(refs_batch,cands_batch):
            scores.append(float(j))
  return scores





