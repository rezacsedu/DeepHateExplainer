import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertModel
import numpy as np
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


class BERTBengali(nn.Module):
    def __init__(self):
        super(BERTBengali, self).__init__()
        #self.bert = BertForMaskedLM.from_pretrained("sagorsarker/bangla-bert-base")
        self.bert = BertModel.from_pretrained(args.pretrained_model_name)
        self.bert_drop = nn.Dropout(args.dropout)
        self.out = nn.Linear(args.bert_hidden, args.classes)

    def forward(self, ids, attention_mask, token_type_ids):
        _,o2 = self.bert(
            ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

class CustomBERTBengali(nn.Module):
    def __init__(self):
        super(CustomBERTBengali, self).__init__()
        #self.bert = BertForMaskedLM.from_pretrained("sagorsarker/bangla-bert-base") 
        self.bert = BertModel.from_pretrained(args.pretrained_model_name)
        self.bert_drop = nn.Dropout(args.dropout) #0.3
        self.out = nn.Linear(args.bert_hidden * 2, args.classes)

    def forward(self, ids, attention_mask, token_type_ids):
        o1,o2 = self.bert(
            ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)
        bo = self.bert_drop(cat)
        logits = self.out(bo)       
        return logits

class BERTBengaliTwo(nn.Module):
    def __init__(self):
        super(BERTBengaliTwo, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        self.drop_out = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.bert_hidden * 2, args.classes)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, attention_mask, token_type_ids):
        _, _, out = self.bert(
            ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        out = out[:,0,:]
        logits = self.l0(out)
        return logits


class BERTBengaliNext(nn.Module):
    def __init__(self):
        super(BERTBengaliNext, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        self.drop_out = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.bert_hidden * 4, args.classes)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, args.bert_hidden)

    def forward(self,ids,attention_mask, token_type_ids):
        _, _, hidden_states = self.bert(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        vec1 = self._get_cls_vec(hidden_states[-1])
        vec2 = self._get_cls_vec(hidden_states[-2])
        vec3 = self._get_cls_vec(hidden_states[-3])
        vec4 = self._get_cls_vec(hidden_states[-4])

        out = torch.cat([vec1, vec2, vec3, vec4], dim=1)
        out = self.drop_out(out)
        logits = self.l0(out)
        return logits
