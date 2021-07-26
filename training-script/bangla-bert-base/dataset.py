import utils
import pandas as pd 
import torch
import numpy as np
import warnings
from transformers import BertTokenizer,AutoTokenizer
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
warnings.filterwarnings("ignore")

class BengaliDataset:
    def __init__(self, text, targets):
        self.text = text 
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
        self.max_length = args.max_len
        self.targets = targets

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation_strategy="longest_first",
            pad_to_max_length=True,
            truncation=True
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
       
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }


if __name__ == "__main__":
    print(args.training_file)
    df = pd.read_csv(args.training_file).dropna().reset_index(drop = True)
    print(set(df['label'].values))
    dset = BengaliDataset(
        text=df.text.values,
        targets=df.target.values
        )
    print(df.iloc[1]['text'])
    print(dset[1])
