from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch



class Bleurt(torch.nn.Module):
    def __init__(self,model_type,device):
        super(Bleurt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_type)
        self.model.to(device)
        self.device = device
    def padding(self, squence, max_length, pad_token=0):
        padding_length = max_length - len(squence)
        return squence + [pad_token] * padding_length

    def forward(self, refs, hyps):
        tokens = self.tokenizer(refs, hyps,add_special_tokens=True, max_length=512,  
        truncation=True)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        token_type_ids = tokens['token_type_ids']


        for i in range(len(input_ids)):
            input_ids[i] = self.padding(input_ids[i], 512, pad_token=0)
            attention_mask[i] = self.padding(attention_mask[i], 512, pad_token=0)
            token_type_ids[i] = self.padding(token_type_ids[i], 512, pad_token=0)

        input_ids = torch.tensor(input_ids,device=self.device)
        attention_mask = torch.tensor(attention_mask,device=self.device)
        token_type_ids = torch.tensor(token_type_ids,device=self.device)
        scores = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0].squeeze()
        


        return {
            'score': scores,
        }
