import utils
import dataset
import engine
import torch
import transformers
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from settings import get_module_logger
from model import BERTBengali, BERTBengaliTwo, BERTBengaliNext, CustomBERTBengali
from sklearn import model_selection
from transformers import AdamW
from vis import display_acc_curves, display_loss_curves
from dataset import BengaliDataset
from transformers import get_linear_schedule_with_warmup
from vis import display_acc_curves, display_loss_curves
from collections import defaultdict
from prediction import get_predictions
from test_eval import test_evaluation
import gc
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)


def run():

    df_train = pd.read_csv(args.training_file).dropna().reset_index(drop=True)
    df_valid = pd.read_csv(args.validation_file).dropna().reset_index(drop=True)

    logger.info("train len - {} valid len - {}".format(len(df_train), len(df_valid)))
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    #df_train = df_train.sample(frac=1).reset_index(drop=True)

    train_dataset = BengaliDataset(
        text=df_train.text.values,
        targets=df_train.target.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = BengaliDataset(
        text=df_valid.text.values,
        targets=df_valid.target.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_layer == "last_two":
        model = BERTBengaliTwo()
    elif args.model_layer == "last_four":
        model = BERTBengaliNext()
    elif args.model_layer == "custom":
        model = CustomBERTBengali()
    else:
        print("hello")
        model = BERTBengali()
    model.to(device)
   

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / args.train_batch_size * args.epochs)

    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_train_steps
    )
    print("STARTING TRAINING ...\n")
    logger.info("{} - {}".format("STARTING TRAINING",args.model_specification))
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch + 1}/{args.epochs}')
        logger.info('-' * 10)

        train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, len(df_train))
        logger.info(f'Train loss {train_loss} accuracy {train_acc}')
        y_pred, val_loss = engine.eval_fn(valid_data_loader, model, device, len(df_valid))
        val_f1 = f1_score(y_pred, df_valid['target'].values, average='weighted')
        val_f1 = np.round(val_f1.item(), 3)
        logger.info(f'Val loss {val_loss} Val acc {val_f1}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_f1)
        history['val_loss'].append(val_loss)
        if val_f1 > best_accuracy:
            #torch.save(model.state_dict(), f"{args.model_path}{args.model_specification}.bin")
            best_accuracy = val_f1
            test_evaluation(model, device)
    display_acc_curves(history, "acc_curves")
    display_loss_curves(history, "loss_curves")
    del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info("##################################### Task End ############################################")

if __name__ == "__main__":
    run()