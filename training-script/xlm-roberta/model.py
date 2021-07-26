import torch
import torch.nn as nn
import transformers
from transformers import XLMRobertaModel
import numpy as np
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


class RobertaBengali(nn.Module):
    def __init__(self):
        super(RobertaBengali, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(args.pretrained_model_name)
        self.roberta_drop = nn.Dropout(args.dropout) #0.3
        self.out = nn.Linear(args.roberta_hidden, args.classes)

    def forward(self, ids, attention_mask):
        _,o2 = self.roberta(
            ids, 
            attention_mask=attention_mask
        )
        bo = self.roberta_drop(o2)
        output = self.out(bo)
        return output

class CustomRobertaBengali(nn.Module):
    def __init__(self):
        super(CustomRobertaBengali, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(args.pretrained_model_name)
        self.roberta_drop = nn.Dropout(args.dropout) #0.3
        self.out = nn.Linear(args.roberta_hidden * 2, args.classes)

    def forward(self, ids, attention_mask):
        o1,o2 = self.roberta(
            ids, 
            attention_mask=attention_mask
        )
        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)
        bo = self.roberta_drop(cat)
        logits = self.out(bo)       
        return logits

class RobertaBengaliTwo(nn.Module):
    def __init__(self):
        super(RobertaBengaliTwo, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        self.roberta_drop = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.roberta_hidden * 2, args.classes)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, attention_mask):
        _, _, out = self.roberta(
            ids, 
            attention_mask=attention_mask
        )
        out = torch.cat((out[-1], out[-2]), dim=-1)
        #out = self.roberta_drop(out)
        out = out[:,0,:]
        logits = self.l0(out)
        return logits


class RobertaBengaliNext(nn.Module):
    def __init__(self):
        super(RobertaBengaliNext, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        self.roberta_drop = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.roberta_hidden * 4, args.classes)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, args.roberta_hidden)

    def forward(self,ids,attention_mask):
        _, _, hidden_states = self.roberta(
            ids,
            attention_mask=attention_mask
        )
        vec1 = self._get_cls_vec(hidden_states[-1])
        vec2 = self._get_cls_vec(hidden_states[-2])
        vec3 = self._get_cls_vec(hidden_states[-3])
        vec4 = self._get_cls_vec(hidden_states[-4])

        out = torch.cat([vec1, vec2, vec3, vec4], dim=1)
        #out = self.roberta_drop(out)
        logits = self.l0(out)
        return logits
