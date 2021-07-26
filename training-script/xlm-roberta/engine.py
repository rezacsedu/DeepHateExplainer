import utils
import torch
import time
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from settings import get_module_logger
from sklearn.metrics import f1_score
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)


def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)

def train_fn(data_loader, model, optimizer, device, scheduler, n_examples):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    start = time.time()
    train_losses = []
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        # Reset gradients
        model.zero_grad()

        outputs = model(
            ids=ids,
            attention_mask=mask
        )

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())
        outputs = torch.log_softmax(outputs,dim=1)
        outputs = torch.argmax(outputs,dim=1)
        outputs = outputs.cpu().detach().numpy().tolist()
        targets = targets.cpu().detach().numpy().tolist()
        train_f1 = f1_score(outputs, targets, average='macro')
        end = time.time()
        f1 = np.round(train_f1.item(), 3)
        if (bi % 100 == 0 and bi != 0) or (bi == len(data_loader) - 1) :
            logger.info(f'bi={bi}, Train F1={f1},Train loss={loss.item()}, time={end-start}')
        
        loss.backward() # Calculate gradients based on loss
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Adjust weights based on calculated gradients
        scheduler.step() # Update scheduler
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss = losses.avg)
        fin_targets.extend(targets) 
        fin_outputs.extend(outputs)
    f1 = f1_score(fin_outputs, fin_targets, average='weighted')
    f1 = np.round(f1.item(), 3)
    return f1, np.mean(train_losses)


def eval_fn(data_loader, model, device, n_examples):
        model.eval()
        start = time.time()
        losses = utils.AverageMeter()
        val_losses = []
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader))
            for bi, d in enumerate(data_loader):
                ids = d["ids"]
                mask = d["mask"]
                targets = d["targets"]
                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.long)

                outputs = model(
                    ids=ids,
                    attention_mask=mask
                )
                loss = loss_fn(outputs, targets)
                outputs = torch.log_softmax(outputs,dim=1)
                outputs = torch.argmax(outputs,dim=1)
                val_losses.append(loss.item())
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            # test_f1 = f1_score(fin_outputs, fin_targets, average='macro')
            # f1 = np.round(test_f1.item(), 3)
        return fin_outputs, np.mean(val_losses)